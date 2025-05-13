import sys, os, json
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utils'))
from transformers import EarlyStoppingCallback
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
        self.best_idx = None  # 初始化 best_idx

    def fine_tune(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=500,
            weight_decay=0.05,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            learning_rate=self.config['lr'][0]
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset
        )

        trainer.train()

        # Save the model and tokenizer
        self.model.save_pretrained('./model')
        self.tokenizer.save_pretrained('./model')

        # 假设你有一个学习率列表
        lr_list = [0.01,0.001]  # 你可以根据实际情况调整这个列表
        eval_losses = []

        for lr in lr_list:
            trainer.args.learning_rate = lr
            eval_result = trainer.evaluate()
            eval_losses.append(eval_result['eval_loss'])

        # 选择验证损失最小的学习率索引作为 best_idx
        self.best_idx = eval_losses.index(min(eval_losses))

    def query(self, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.config['max_length'], padding='max_length',
                                truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入张量移动到相同的设备
        self.model.to(device)
        outputs = self.model(**inputs)
        prediction = outputs.logits.detach().cpu().numpy()  # 将输出张量移动到CPU以便后续处理
        return prediction[0][0]

    def eval(self, test_texts, test_labels, plot=False, X_grid=None, grid_texts=None, y_grid=None, file_name=None):
        valid_mean = np.mean(self.train_labels)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        y_test_outputs = [self.query(text) for text in test_texts]
        valid_test_y = test_labels
        rmse = RMSE(y_test_outputs, valid_test_y)
        rmse_woo, num_o = RMSE_woo(y_test_outputs, valid_test_y)
        r2 = r2_score(valid_test_y, y_test_outputs)  # 计算 R²

        if plot and X_grid is not None and grid_texts is not None:
            y_grid_outputs = [self.query(text) for text in grid_texts]
            valid_plot_x = np.array([X_grid[i, 0] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])
            valid_plot_y = [y_grid[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None]
            valid_plot_y_outputs = np.array(
                [y_grid_outputs[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])

            plt.scatter(valid_plot_x, valid_plot_y_outputs, c=['b'] * len(valid_plot_x), label='Predicted Labels')
            plt.plot(valid_plot_x, valid_plot_y, c='g', label='True Labels')

            plt.legend()
            plt.title('1D_visualization\n' \
                      + 'RMSE = {:.3f}\n'.format(rmse) \
                      + 'RMSE(woo) = {:.3f}   #outlier: {}\n'.format(rmse_woo, num_o) \
                      + 'R² = {:.3f}'.format(r2))  # 添加 R² 信息
            plt.xlabel('x')
            plt.ylabel('y')

            if file_name is None:
                plt.savefig('plot.png', bbox_inches='tight', dpi=300)
            else:
                plt.savefig(file_name, bbox_inches='tight', dpi=300)
        else:
            y_grid_outputs = None

        print(f'R²: {r2}')  # 打印 R² 值

        return y_test_outputs, y_grid_outputs, len(valid_test_y), rmse, rmse_woo

    def get_eval_losses(self, trainer, lr_list):
        eval_losses = []
        for lr in lr_list:
            trainer.args.learning_rate = lr
            eval_result = trainer.evaluate()
            eval_losses.append(eval_result['eval_loss'])
        return eval_losses

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

def run_setting(data_dir, n_sims=1, num_epochs=1, batch_size=16,
                     data_list=['asset'],
                     lr_list=[0.01, 0.0001],
                     prefix_list=['_', '_fn_'],
                     pc_list=[''],
                     model_name='bge-large-en-v1.5',
                     valid_temperature=0.75):
    config = {'model_name': model_name, "num_epochs": num_epochs, "batch_size": batch_size, "max_length": 256}
    counter = 0

    for sim_idx in range(n_sims):
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        data_sim_dir = '%s/data_%d' % (data_dir, sim_idx + 1)
        if not os.path.isdir(data_sim_dir):
            os.mkdir(data_sim_dir)

    for data in data_list:
        for prefix in prefix_list:
            data_prefix = data.lower() + prefix
            for pc in pc_list:
                print("------------------Running group %d---------------------" % counter)
                print("%s%s" % (data_prefix, pc))
                train_df = pd.read_csv("data/%s/%s_train1%s.csv" % (data, data.lower(), pc))
                valid_df = pd.read_csv("data/%s/%s_valid1.csv" % (data, data.lower()))
                cols = train_df.columns.tolist()
                cols[-1] = 'y'
                cols[:-1] = list(range(len(cols) - 1))
                train_df.columns = cols
                valid_df.columns = cols

                train_texts, train_labels = df2datasets(train_df, label=True)
                valid_texts, valid_labels = df2datasets(valid_df, label=True)

                train_file = "data/%s/%s%strain1.jsonl" % (data, data_prefix, pc)
                valid_file = "data/%s/%svalid1.jsonl" % (data, data_prefix)

                config['lr'] = lr_list
                for sim_idx in range(n_sims):
                    print('---Simulation %d---' % (sim_idx + 1))
                    data_sim_dir = '%s/data_%d' % (data_dir, sim_idx + 1)

                    fine_tuner = FineTuner(config=config, train_texts=train_texts, train_labels=train_labels,
                                           valid_texts=valid_texts, valid_labels=valid_labels)
                    fine_tuner.fine_tune()

                    plot_save_path = valid_file.split('valid.')[0] + ".png"
                    file_name = valid_file.split('valid.')[0].replace(",", "").replace("(", "").replace(")", "") + 'ft_info.json'

                    if sim_idx == 0: config['lr'] = [lr_list[fine_tuner.best_idx]]
                    with open('%s/data_%d/%s%s_ft_info.json' % (data_dir, sim_idx + 1, data, pc), 'w') as fp:
                        json.dump({'model_id': 'bge-large-en-v1.5'}, fp, indent=4)

                    tr_ts_vl_json = {
                        "train_x": train_df[train_df.columns[:-1]].astype(float).values.tolist(),
                        "train_y": list(map(float, train_df['y'])),
                        "validation_x": valid_df[valid_df.columns[:-1]].astype(float).values.tolist(),
                        "validation_y": list(map(float, valid_df['y'])),
                    }

                    with open('%s/data_%d/%s%s_all.json' % (data_dir, sim_idx + 1, data_prefix, pc), 'w') as fp:
                        json.dump(tr_ts_vl_json, fp)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data_dir = "./bge_data"  # 设置数据目录
    run_setting(data_dir)
