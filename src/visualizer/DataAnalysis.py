import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimSun']#默认中文字体宋体
plt.rcParams['axes.unicode_minus'] = False #正常显示负号

class Draw:
    def __init__(self,
                 data,
                 labelcolname, #data中标签列的列名
                 labelmap, #{0:"无岩爆",1:"轻微岩爆",2:"中等岩爆",3:"强岩爆"}

                 ):
        colormap = ["#FFB3DD", "#FDC1C1", "#F8E7C0", "#E1F8C0", "#9ABBF3", "#F7786D", "#B4ECF0", "#9C9396"]
        if len(labelmap) <= len(colormap):
            self.colormap = colormap[:len(labelmap)]
        else:
            self.colormap = [colormap[i % len(colormap)] for i in range(len(labelmap))]
        self.data = data
        self.labelmap = labelmap
        self.colsname = [col for col in data.columns if col != labelcolname]
        self.labelcolname = labelcolname

    def plotpie(self, show=True, save=True):  #画饼图
        label_counts = Counter(self.data[self.labelcolname])
        count_dict = dict(sorted(label_counts.items()))
        unique_elements = list(count_dict.keys())
        sizes = [count_dict[k] for k in unique_elements]
        legend_labels = [self.labelmap[k] for k in unique_elements]
        plt.pie(sizes, colors=self.colormap, autopct="%1.1f%%", wedgeprops={"edgecolor": "white", 'width': 0.6},
                textprops={'fontname': 'Times New Roman', 'fontsize': 12})
        plt.legend(
            legend_labels,
            prop={'family': 'SimSun', 'size': 12},
            loc="upper right",
            bbox_to_anchor=(1.25, 0.90),
            frameon=False)
        if save:
            plt.savefig("./pie.tif", dpi=300)
        if show:
            plt.show()

    def plotbox(self, show=True, save=True):
        unique_labels = sorted(self.labelmap.keys())
        plt.figure(figsize=(10, 6))
        for col_idx, col in enumerate(self.colsname):
            groups = self.data.groupby(self.labelcolname)
            data_to_plot = []
            for label in unique_labels:
                data_to_plot.append(groups.get_group(label)[col].dropna().values)

            box = plt.boxplot(
                data_to_plot,
                positions=[i + 1 for i in range(len(unique_labels))],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', linewidth=1.5)
            )
            for patch, color in zip(box['boxes'], self.colormap):
                patch.set_facecolor(color)
                patch.set_edgecolor('black')
                patch.set_alpha(0.8)

            for i, (label, color) in enumerate(zip(unique_labels, self.colormap)): #散点
                if label in groups.groups:
                    group_data = groups.get_group(label)[col].dropna()
                    if not group_data.empty:
                        x_data = i + 1
                        y_data = group_data.values
                        plt.scatter(
                            x_data,
                            y_data,
                            color=color,
                            alpha=0.6,
                            edgecolor='k',
                            s=30,
                            zorder=2
                        )
                        mean_val = group_data.mean()
                        plt.hlines(
                            mean_val,
                            xmin=x_data - 0.15,
                            xmax=x_data + 0.15,
                            colors='red',
                            linestyles='-',
                            linewidth=2,
                            zorder=3
                        )
                        plt.text( #标均值
                            x_data + 0.18,
                            mean_val,
                            f'{mean_val:.2f}',
                            color='red',
                            fontsize=10,
                            verticalalignment='center'
                        )

        # 设置 X 轴刻度标签
        plt.xticks(
            ticks=[i + 1 for i in range(len(unique_labels))],
            labels=[self.labelmap[l] for l in unique_labels],
            fontsize=10
        )
        if save:
            plt.savefig("./boxscatter.tif", dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    def calc_stats(self, save=True): #统计参数
        stats_list = []
        for col in self.colsname:
            series = self.data[col].dropna()
            n_valid = len(series)
            if n_valid == 0:
                continue
            mean_val = series.mean()
            std_val = series.std()
            var_val = series.var()
            median_val = series.median()
            skew_val = series.skew()
            kurt_val = series.kurtosis()
            cv_val = (std_val / mean_val) if mean_val != 0 else 0.0
            stat_dict = {
                '参数名': col,
                '均值': round(mean_val, 2),
                '标准差': round(std_val, 2),
                '方差': round(var_val, 2),
                '中位数': round(median_val, 2),
                '最小值': round(series.min(), 2),
                '最大值': round(series.max(), 2),
                '偏度 (Skew)': round(skew_val, 2),
                '峰度 (Kurt)': round(kurt_val, 2),
                '变异系数 (CV)': round(cv_val, 2)
            }
            stats_list.append(stat_dict)
        stats_df = pd.DataFrame(stats_list)
        if save:
            csv_path = "./statistics_summary.csv"
            stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return stats_df



if __name__ == "__main__":
    data = pd.read_csv('F:\pycharm\Python_Project\RockBrust\static\merge.csv')
    data['MR'] = data['MR'].astype(int)
    draw = Draw(data, 'MR', {0:"无岩爆",1:"轻微岩爆",2:"中等岩爆",3:"强岩爆"})
    draw.plotpie()
    draw.plotbox()
    draw.calc_stats()
















