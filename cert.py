from train import train
from pred import predict
import os

def cert(csv_path = "./data/50a_0.3_0_1.0_grande_msa.csv", model_path = "./model_weight/50a_0.3_0_1.0_mutfinder.pt",
         epoch_model = 0, arch_model = (0,1,2), n_msa = 2, n_seq = 25, batch_size = 2,
         heads = 12, d_ffn = 3072, num_msa_blocks = 0, col_bias = True,
         num_epochs = 1000, early_stop = 30, row_dropout = 0.0, column_dropout = 0.0, ffn_dropout = 0.0, lr = 1e-4, decay_rate = 1e-2, l = "l2",
         maxtop = 163, attntop = 5):

    if len(arch_model) >= 1:
        for i in arch_model:
            path, ext = os.path.splitext(model_path)
            new_path = f"{i}arch/{path}{ext}"
            train(
                csv_path=csv_path, model_path=new_path, epoch_model=epoch_model,
                heads=heads, d_ffn=d_ffn, num_msa_blocks=i, col_bias=col_bias,
                row_dropout=row_dropout, column_dropout=column_dropout, ffn_dropout=ffn_dropout,
                num_epochs=num_epochs, early_stop=early_stop,
                lr=lr, decay_rate=decay_rate, l=l,
                n_msa=n_msa, n_seq=n_seq, batch_size=batch_size)
    else:
        train(
            csv_path=csv_path, model_path=model_path, epoch_model=epoch_model,
            heads=heads, d_ffn=d_ffn, num_msa_blocks=num_msa_blocks, col_bias=col_bias,
            row_dropout=row_dropout, column_dropout=column_dropout, ffn_dropout=ffn_dropout,
            num_epochs=num_epochs, early_stop=early_stop,
            lr=lr, decay_rate=decay_rate, l=l,
            n_msa=n_msa, n_seq=n_seq, batch_size=batch_size)

    if len(arch_model) >= 1:
        for i in arch_model:
            path, ext = os.path.splitext(model_path)
            new_path = f"{i}arch/{path}{ext}"
            if epoch_model > 0:
                # 读取最佳模型
                predict(csv_path=csv_path, model_path=new_path,
                        heads=heads, d_ffn=d_ffn, num_msa_blocks=i, col_bias=col_bias,
                        maxtop=maxtop, attntop=attntop)
                # 读取循环模型
                for j in [epoch for epoch in range(num_epochs) if (epoch + 1) // (num_epochs // epoch_model)]:
                    ori_path, ext = os.path.splitext(new_path)
                    new_path = f"{ori_path}{j}{ext}"
                    predict(csv_path=csv_path, model_path=new_path,
                            heads=heads, d_ffn=d_ffn, num_msa_blocks=i, col_bias=col_bias,
                            maxtop=maxtop, attntop=attntop)
            else:
                predict(csv_path=csv_path, model_path=new_path,
                        heads=heads, d_ffn=d_ffn, num_msa_blocks=i, col_bias=col_bias,
                        maxtop=maxtop, attntop=attntop)
    else:
        if epoch_model > 0:
            predict(csv_path=csv_path, model_path=model_path,
                    heads=heads, d_ffn=d_ffn, num_msa_blocks=num_msa_blocks, col_bias=col_bias,
                    maxtop=maxtop, attntop=attntop)
            for i in [epoch for epoch in range(num_epochs) if (epoch + 1) // (num_epochs // epoch_model)]:
                ori_path, ext = os.path.splitext(model_path)
                new_path = f"{ori_path}{i}{ext}"
                predict(csv_path=csv_path, model_path=new_path,
                        heads=heads, d_ffn=d_ffn, num_msa_blocks=num_msa_blocks, col_bias=col_bias,
                        maxtop=maxtop, attntop=attntop)
        else:
            predict(csv_path=csv_path, model_path=model_path,
                    heads=heads, d_ffn=d_ffn, num_msa_blocks=num_msa_blocks, col_bias=col_bias,
                    maxtop=maxtop, attntop=attntop)


if __name__ == '__main__':
    cert(csv_path = "./data/50a_0.3_0_1.0_grande_msa.csv",
         model_path = "./model_weight/50a_0.3_0_1.0_mutfinder.pt",

         n_msa=2,
         n_seq=25,
         batch_size=2,

         epoch_model = 0,
         arch_model = (0,1,2),
         num_msa_blocks=0,
         heads = 12,
         d_ffn = 3072,
         col_bias = True,

         num_epochs = 1000,
         early_stop = 30,
         row_dropout = 0.0,
         column_dropout = 0.0,
         ffn_dropout = 0.0,
         lr = 1e-4,
         decay_rate = 1e-2,
         l = "l2",

         maxtop = 163,
         attntop = 5
         )

