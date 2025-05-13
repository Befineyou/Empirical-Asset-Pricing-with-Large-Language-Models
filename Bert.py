import sys, os, json
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utils'))
from transformers import EarlyStoppingCallback
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from functools import partial
from sklearn.feature_selection import mutual_info_regression, f_classif
from scipy.stats import pearsonr
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")


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

class BertFineTuner(object):
    def __init__(self, config, train_texts, train_labels, valid_texts, valid_labels):
        self.config = config
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.valid_texts = valid_texts
        self.valid_labels = valid_labels
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        self.model = BertForSequenceClassification.from_pretrained(config['model_name'], num_labels=1)
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
        lr_list = [0.000001]  # 你可以根据实际情况调整这个列表
        eval_losses = []

        for lr in lr_list:
            trainer.args.learning_rate = lr
            eval_result = trainer.evaluate()
            eval_losses.append(eval_result['eval_loss'])

        # 选择验证损失最小的学习率索引作为 best_idx
        self.best_idx = eval_losses.index(min(eval_losses))

    # def query(self, text):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     inputs = self.tokenizer(text, return_tensors='pt', max_length=self.config['max_length'], padding='max_length',
    #                             truncation=True)
    #     self.model.to(device)
    #     outputs = self.model(**inputs)
    #     prediction = outputs.logits.detach().numpy()
    #     return prediction[0][0]

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

        # 计算最佳学习率索引并赋值给 best_idx
        #self.best_idx = self.calculate_best_lr_idx()

        if plot and X_grid is not None and grid_texts is not None:
            y_grid_outputs = [self.query(text) for text in grid_texts]
            valid_plot_x = np.array([X_grid[i, 0] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])
            valid_plot_y = [y_grid[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None]
            valid_plot_y_outputs = np.array(
                [y_grid_outputs[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])

            plt.scatter(valid_plot_x, valid_plot_y_outputs, c=['b'] * len(valid_plot_x), label='Bert Predicted Labels')
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




def data2text(row,  label ,integer=False):
    prompt = "When we have "

    for i in range(1, len(row) - True):
        if integer:
            prompt += "x%d=%d, " % (i, row[i])
        else:
            prompt += "x%d=%.4f, " % (i, row[i])
    prompt += "what should be the y value?"




    if not label:
        completion = "%.6f" % row['y']

        return "%s###" % prompt , completion
    else:
        if integer:
            completion = "%.6f" % row['y']
        else:
            completion = "%.6f" % row['y']

        return prompt, completion


import json
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, f_classif
from scipy.stats import pearsonr

import json
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, f_classif
from scipy.stats import pearsonr

import json
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, f_classif
from scipy.stats import pearsonr


def df2datasets(train_df, valid_df=None, test_df=None, label=True, integer=False, top_n=20):
    # 将所有列名转换为字符串，以避免因列名为int类型而导致的错误
    train_df.columns = [str(col) for col in train_df.columns]
    if valid_df is not None:
        valid_df.columns = [str(col) for col in valid_df.columns]
    if test_df is not None:
        test_df.columns = [str(col) for col in test_df.columns]

    # 删除 'sic' 开头的列，只保留数值型特征
    numeric_features = [col for col in train_df.columns[:-1] if not col.startswith('sic')]
    target = train_df.columns[-1]  # 目标变量列

    # 确保数值型特征不为空
    if not numeric_features:
        raise ValueError("No numeric features found in the dataset.")

    # 使用训练集的数值特征进行特征选择
    pcc_scores = []
    for feature in numeric_features:
        score, _ = pearsonr(train_df[feature], train_df[target])
        pcc_scores.append(abs(score))
    pcc_ranks = pd.Series(pcc_scores, index=numeric_features).rank(ascending=False)

    anova_scores, _ = f_classif(train_df[numeric_features], train_df[target])
    anova_ranks = pd.Series(anova_scores, index=numeric_features).rank(ascending=False)

    mi_scores = mutual_info_regression(train_df[numeric_features], train_df[target])
    mi_ranks = pd.Series(mi_scores, index=numeric_features).rank(ascending=False)

    combined_ranks = (pcc_ranks + anova_ranks + mi_ranks) / 3
    top_features = combined_ranks.nsmallest(top_n).index.tolist()
    important_features = top_features[:5]

    def generate_prompts(df):
        texts, labels = [], []
        for _, row in df.iterrows():
            prompt = "Assume you are a stock market investor. "
            for feature in top_features:
                feature_value = row[feature]
                feature_description = f"{feature}={feature_value:.4f}"
                if feature in important_features:
                    feature_description = f"**particularly important** {feature_description}"
                prompt += f"If a company's {feature_description}, "
            prompt += "then the expected return (RET) of the company's stock is?"
            if label:
                completion = f"{row[target]:.6f}"
                texts.append(f"{prompt}###")
                labels.append(float(completion))
            else:
                texts.append(f"{prompt}###")
        return texts, labels

    # 生成训练集、验证集和测试集的 Prompts 和 labels
    train_texts, train_labels = generate_prompts(train_df)
    valid_texts, valid_labels = generate_prompts(valid_df) if valid_df is not None else ([], [])
    test_texts, test_labels = generate_prompts(test_df) if test_df is not None else ([], [])

    return train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels


def generate_prompts_and_completions(df, top_features, important_features, target, label):
    texts = []
    labels = []
    for _, row in df.iterrows():
        prompt = "Assume you are a stock market investor. "
        for feature in top_features:
            feature_value = row[feature]
            feature_description = f"{feature}={feature_value:.4f}"
            if feature in important_features:
                feature_description = f"**particularly important** {feature_description}"
            prompt += f"If a company's {feature_description}, "
        prompt += "then the expected return (RET) of the company's stock is?"

        if label:
            completion = f"{row[target]:.6f}"
            texts.append(f"{prompt}###")
            labels.append(float(completion))
        else:
            texts.append(f"{prompt}###")

    return texts, labels


def run_setting_bert(data_dir, n_sims=2, num_epochs=2, batch_size=32,
                     data_list=['asset'],
                     lr_list=[0.0001],
                     prefix_list=['_', '_fn_'],
                     pc_list=[''],
                     model_name='bert-base-uncased',
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
                test_df = pd.read_csv("data/%s/%s_test1.csv" % (data, data.lower()))
                cols = train_df.columns.tolist()
                cols[-1] = 'y'
                cols[:-1] = list(range(len(cols) - 1))
                train_df.columns = cols
                valid_df.columns = cols
                test_df.columns = cols
                train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels = df2datasets(train_df,
                                                                                                            valid_df,
                                                                                                            test_df,
                                                                                                            label=True)
                # train_texts, train_labels = df2datasets(train_df,label=True)
                # valid_texts, valid_labels = df2datasets(valid_df,label=True)
                # #test_texts, test_labels = df2datasets(test_df, label=False)



                train_file = "data/%s/%s%strain11.jsonl" % (data, data_prefix, pc)
                valid_file = "data/%s/%svalid11.jsonl" % (data, data_prefix)

                train_json = [{"prompt": t, "completion": c} for t, c in zip(train_texts, train_labels)]
                valid_json = [{"prompt": t, "completion": c} for t, c in
                              zip(valid_texts, valid_labels)] if valid_texts else None
                test_json = [{"prompt": t, "completion": c} for t, c in
                             zip(test_texts, test_labels)] if test_texts else None

                with open("train_prompts.json", "w") as f:
                    json.dump(train_json, f, indent=4)

                with open("valid_prompts.json", "w") as f:
                    json.dump(valid_json, f, indent=4)




                config['lr'] = lr_list
                for sim_idx in range(n_sims):
                    print('---Simulation %d---' % (sim_idx + 1))
                    data_sim_dir = '%s/data_%d' % (data_dir, sim_idx + 1)

                    bert_fine_tuner = BertFineTuner(config=config, train_texts=train_texts, train_labels=train_labels,
                                                    valid_texts=valid_texts, valid_labels=valid_labels)
                    bert_fine_tuner.fine_tune()

                    plot_save_path = valid_file.split('valid.')[0] + ".png"
                    file_name = valid_file.split('valid.')[0].replace(",", "").replace("(", "").replace(")",
                                                                                                        "") + 'ft_info.json'
                    # y_test_outputs, y_grid_outputs, _, _, _ = bert_fine_tuner.eval(test_texts=test_texts,
                    #                                                                test_labels=test_labels,
                    #                                                                plot=False, file_name=plot_save_path)
                    #
                    # results_df = pd.DataFrame({'True Values': test_labels, 'Predicted Values': y_test_outputs})
                    # results_df.to_csv('predictions_vs_true.csv', index=False)
                    #
                    # # 可视化预测值和真实值
                    # plt.figure(figsize=(10, 6))
                    # plt.scatter(range(len(test_labels)), test_labels, color='blue', label='True Values')
                    # plt.scatter(range(len(test_labels)), y_test_outputs, color='red', label='Predicted Values')
                    # plt.xlabel('Samples')
                    # plt.ylabel('Values')
                    # plt.title('True Values vs Predicted Values')
                    # plt.legend()
                    # plt.savefig('predictions_vs_true.png', bbox_inches='tight', dpi=300)
                    # plt.show()


                    if sim_idx == 0: config['lr'] = [lr_list[bert_fine_tuner.best_idx]]
                    # save fine-tuned info and results
                    with open('%s/data_%d/%s%s_ft_info.json' % (data_dir, sim_idx + 1, data, pc), 'w') as fp:
                        json.dump({'model_id': 'bert'}, fp, indent=4)

                    # tr_ts_vl_json = {"train_x": train_df[train_df.columns[:-1]].values.tolist(),
                    #                  "train_y": list(train_df['y']),
                    #                  "validation_x": valid_df[valid_df.columns[:-1]].values.tolist(),
                    #                  "validation_y": list(valid_df['y']),
                    #                  "test_x": test_df[test_df.columns[:-1]].values.tolist(),
                    #                  "test_y": list(test_df['y']), "bert_test_y": y_test_outputs}
                    # with open('%s/data_%d/%s%s_all.json' % (data_dir, sim_idx + 1, data_prefix, pc), 'w') as fp:
                    #     json.dump(tr_ts_vl_json, fp)
                    tr_ts_vl_json = {
                        "train_x": train_df[train_df.columns[:-1]].astype(float).values.tolist(),
                        "train_y": list(map(float, train_df['y'])),
                        "validation_x": valid_df[valid_df.columns[:-1]].astype(float).values.tolist(),
                        "validation_y": list(map(float, valid_df['y'])),
                        # "test_x": test_df[test_df.columns[:-1]].astype(float).values.tolist(),
                        # "test_y": list(map(float, test_df['y'])),
                        # "bert_test_y": list(map(float, y_test_outputs))
                    }

                    with open('%s/data_%d/%s%s_all.json' % (data_dir, sim_idx + 1, data_prefix, pc), 'w') as fp:
                        json.dump(tr_ts_vl_json, fp)


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_dir = "./bert_data"  # 设置数据目录
    run_setting_bert(data_dir)
