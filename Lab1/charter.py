import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob, os
import numpy as np


def prepare_upper_x_axis(plot, xUpperAxis):
    upperX = plot.twiny()
    upperX.set_xlim(plot.get_xlim())
    upperX.set_xlabel("Pokolenie")
    xUpperAxis.append(200)
    upperX.set_xticks(np.array(xUpperAxis[0:201:40]))


def prepare_bottom_x_axis(plot):
    plot.set_xlabel("Rozegranych gier (x 1000)")
    xAxisFormatter = ticker.FuncFormatter(lambda x, pos: str(int(x / 1000)))
    plot.xaxis.set_major_formatter(xAxisFormatter)


def prepare_y_axis(plot):
    yAxisFormatter = ticker.FuncFormatter(lambda x, pos: str(int(x * 100)))
    plot.yaxis.set_major_formatter(yAxisFormatter)
    plot.yaxis.set_tick_params()
    plot.set_yticks([x / 100 for x in range(60, 101, 5)])


def limit_y_axis(plot):
    plot.set_ylim(0.6, 1)


def limit_chart(plot):
    plot.set_xlim(0, 500000)
    limit_y_axis(plot)


def style_grid(plot):
    plot.yaxis.grid(True)
    plot.yaxis.grid(True)
    plot.grid(linestyle='--')


def prepare_line_chart(lineChart, data, xAxis, xUpperAxis):
    lineChart.set_ylabel("Odsetek wygranych gier [%]")
    prepare_upper_x_axis(lineChart, xUpperAxis)
    for d in data:
        current_data = data[d]
        markers_on = range(0, len(current_data['avg_data']))[0::20]
        lineChart.plot(xAxis, current_data["avg_data"], label=current_data['label'], marker=current_data['marker'],
                       markevery=markers_on, color=current_data['color'], linewidth=0.75, markeredgecolor="black")
    prepare_bottom_x_axis(lineChart)
    prepare_y_axis(lineChart)
    limit_chart(lineChart)
    style_grid(lineChart)
    lineChart.legend(numpoints=2)


def prepare_whiskey_chart(whiskey_chart, whiskey_chart_data):
    box_plot_data = []
    box_plot_labels = []
    for d in whiskey_chart_data:
        current_data = data[d]
        box_plot_data.append(current_data['last_row'])
        box_plot_labels.append(current_data['label'])
    boxplot = whiskey_chart.boxplot(box_plot_data, 1, 'b+', labels=box_plot_labels, vert=True,
                                    showmeans=True, whis=1.5)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(boxplot[element], color="blue")
    plt.setp(boxplot["medians"], color="red")
    plt.setp(boxplot["caps"], color="black", linestyle="-")
    plt.setp(boxplot["whiskers"], color="blue", linestyle="-.")
    plt.setp(boxplot["means"], markerfacecolor="blue", marker='o', markeredgecolor="black")

    prepare_y_axis(whiskey_chart)
    limit_y_axis(whiskey_chart)
    whiskey_chart.yaxis.tick_right()
    for label in whiskey_chart.get_xmajorticklabels():
        label.set_rotation(20)
    style_grid(whiskey_chart)


data = {
    "rsel.csv": {"id": 0, "label": "1-Evol-RS", "marker": "o", "color":"blue"},
    "cel-rs.csv": {"id": 1, "label": "1-Coev-RS", "marker": "v", "color": "green"},
    "2cel-rs.csv": {"id": 2, "label": "2-Coev-RS", "marker": "D", "color": "red"},
    "cel.csv": {"id": 3, "label": "1-Coev", "marker": "s", "color": "black"},
    "2cel.csv": {"id": 4, "label": "2-Coev", "marker": "d", "color": "magenta"}
}


def main():
    os.chdir("./")
    files = glob.glob("*.csv")
    figure = plt.figure(figsize=(10, 7), dpi=80)
    lineChart = figure.add_subplot(121)
    whiskeyChart = figure.add_subplot(122)
    computed = False

    for file in files:
        chartData = np.genfromtxt(file, delimiter=',', skip_header=1)
        if not computed:
            xAxis = [i[1] for i in chartData]
            xUpperAxis = [i[0] for i in chartData]
            computed = True

        avg_data = [(sum(i[2:]) / len(i[2:])) for i in chartData]
        data[file]["avg_data"] = avg_data
        data[file]['last_row'] = [i for i in chartData[-1]][2:]
        print(len(data[file]['last_row']))
    prepare_line_chart(lineChart, data, xAxis, xUpperAxis)
    prepare_whiskey_chart(whiskeyChart, data)

    plt.legend()
    plt.show()
#
if __name__ == "__main__":
    main()