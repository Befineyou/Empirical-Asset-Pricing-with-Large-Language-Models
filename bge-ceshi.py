import sys, os, json
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


def L2error(y1, y2):
    return np.linalg.norm(np.array(y1) - np.array(y2))


def RMSE(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('RMSE input error')
    return np.mean((a - b) ** 2) ** 0.5


def RMSE_woo(a, b, threshold=20):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('RMSE input error')
    std = RMSE(a, b)
    outlier_flag = (np.abs(a - b) > std * threshold)
    num_outlier = np.sum(outlier_flag)

    return RMSE(a[~outlier_flag], b[~outlier_flag]), num_outlier


def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    score = 1 - (numerator / denominator)
    return score


class FineTuner(object):
    def __init__(self, config, train_texts, train_labels, valid_texts, valid_labels):
        self.config = config
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.valid_texts = valid_texts
        self.valid_labels = valid_labels
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=1)
        self.train_dataset = CustomDataset(train_texts, train_labels, self.tokenizer, config['max_length'])
        self.valid_dataset = CustomDataset(valid_texts, valid_labels, self.tokenizer, config['max_length'])
        self.best_idx = None

    def fine_tune(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset
        )

        trainer.train()

        self.model.save_pretrained('./model')
        self.tokenizer.save_pretrained('./model')

        lr_list = [0.001, 0.0001, 0.00001]
        eval_losses = []

        for lr in lr_list:
            trainer.args.learning_rate = lr
            eval_result = trainer.evaluate()
            eval_losses.append(eval_result['eval_loss'])

        self.best_idx = eval_losses.index(min(eval_losses))

    def query(self, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.config['max_length'], padding='max_length',
                                truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        self.model.to(device)
        outputs = self.model(**inputs)
        prediction = outputs.logits.detach().cpu().numpy()
        return prediction[0][0]

    def eval(self, test_texts, test_labels, plot=False, X_grid=None, grid_texts=None, y_grid=None, file_name=None):
        valid_mean = np.mean(self.train_labels)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        y_test_outputs = [self.query(text) for text in test_texts]
        valid_test_y = test_labels
        rmse = RMSE(y_test_outputs, valid_test_y)
        rmse_woo, num_o = RMSE_woo(y_test_outputs, valid_test_y)
        r2 = r2_score(valid_test_y, y_test_outputs)

        if plot and X_grid is not None and grid_texts is not None:
            y_grid_outputs = [self.query(text) for text in grid_texts]
            valid_plot_x = np.array([X_grid[i, 0] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])
            valid_plot_y = [y_grid[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None]
            valid_plot_y_outputs = np.array(
                [y_grid_outputs[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])

            plt.scatter(valid_plot_x, valid_plot_y_outputs, c=['b'] * len(valid_plot_x), label='BGE Predicted Labels')
            plt.plot(valid_plot_x, valid_plot_y, c='g', label='True Labels')

            plt.legend()
            plt.title('1D_visualization\n' \
                      + 'RMSE = {:.3f}\n'.format(rmse) \
                      + 'RMSE(woo) = {:.3f}   #outlier: {}\n'.format(rmse_woo, num_o) \
                      + 'R² = {:.3f}'.format(r2))
            plt.xlabel('x')
            plt.ylabel('y')

            if file_name is None:
                plt.savefig('plot.png', bbox_inches='tight', dpi=300)
            else:
                plt.savefig(file_name, bbox_inches='tight', dpi=300)
        else:
            y_grid_outputs = None

        print(f'R²: {r2}')

        return y_test_outputs, y_grid_outputs, len(valid_test_y), rmse, rmse_woo


def data2text(row, label, integer=False):
    prompt = "When we have "

    for i in range(1, len(row) - True):
        if integer:
            prompt += "x%d=%d, " % (i, row[i])
        else:
            prompt += "x%d=%.4f, " % (i, row[i])
    prompt += "what should be the y value?"

    if not label:
        completion = "%.6f" % row['y']
        return "%s###" % prompt, completion
    else:
        if integer:
            completion = "%.6f" % row['y']
        else:
            completion = "%.6f" % row['y']
        return prompt, completion


def df2datasets(df, label, integer=False):
    texts = []
    labels = []
    for _, row in df.iterrows():
        prompt, completion = data2text(row, label, integer)
        texts.append(prompt)
        labels.append(float(completion))
    return texts, labels


# 仅进行测试
def run_test(model_path, test_data_path):
    config = {'model_name': 'bge-large-en-v1.5', "num_epochs": 5, "batch_size": 10, "max_length": 512}

    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # 加载测试数据
    test_df = pd.read_csv(test_data_path)
    cols = test_df.columns.tolist()
    cols[-1] = 'y'
    cols[:-1] = list(range(len(cols) - 1))
    test_df.columns = cols

    test_texts, test_labels = df2datasets(test_df, label=False)

    # 评估模型
    fine_tuner = FineTuner(config=config, train_texts=[], train_labels=[], valid_texts=[], valid_labels=[])
    fine_tuner.model = model
    fine_tuner.tokenizer = tokenizer

    y_test_outputs, _, _, rmse, rmse_woo = fine_tuner.eval(test_texts=test_texts, test_labels=test_labels)

    # 计算 R²
    r2 = r2_score(test_labels, y_test_outputs)

    # 打印评估结果
    print(f'RMSE: {rmse}')
    print(f'RMSE without outliers: {rmse_woo}')
    print(f'R²: {r2}')
    # 输出预测值和真实值到文件
    results_df = pd.DataFrame({'True Values': test_labels, 'Predicted Values': y_test_outputs})
    results_df.to_csv('predictions_vs_true.csv', index=False)

    # 可视化预测值和真实值
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(test_labels)), test_labels, color='blue', label='True Values')
    plt.scatter(range(len(test_labels)), y_test_outputs, color='red', label='Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title('True Values vs Predicted Values')
    plt.legend()
    plt.savefig('predictions_vs_true.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    model_path = "model"  # 模型路径
    test_data_path = "data/asset/asset_test1.csv"  # 测试数据路径
    run_test(model_path, test_data_path)
