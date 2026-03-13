import numpy as np
import torch

class EarlyStopping:

    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1=0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,F1,F2,F3,F4,model,modelname,dataname):

        score = (accs+F1+F2+F3+F4) /5

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.3f}|NR F1: {:.3f}|FR F1: {:.3f}|TR F1: {:.3f}|UR F1: {:.3f}"
                      .format(self.accs,self.F1,self.F2,self.F3,self.F4))
        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.save_checkpoint(val_loss, model,modelname,dataname)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, modelname, dataname):
        '''当综合评分提升时，保存模型'''
        import torch
        # 把模型名和数据集名拼起来当文件名，防止覆盖。比如 GACL_twitter16.pt
        save_path = f"{modelname}_{dataname}.pt"
        print(f"成绩破纪录啦！正在保存极品装备(模型) -> {save_path}")
        # 真正执行写盘保存的操作
        torch.save(model.state_dict(), save_path)
        # 更新底层的 loss 记录
        self.val_loss_min = val_loss