# 头文件
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QColorDialog, QMessageBox
)
from sc_ui import Ui_dialog
import sys
from matplotlib import pyplot as plt
import numpy as np
from PyQt6.QtGui import (QPixmap)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from osgeo import gdal, osr
import os
import threading
import re

data = pd.DataFrame()
data_1 = pd.DataFrame()
data_2 = pd.DataFrame()
x_col = 0
y_col = 1
ylabel = None
xlabel = None
kind = None
axes = None
fig = None
if_show = 0
sub_colums = None
sub_rows = None
ndvi_arr = pd.DataFrame()
tag, tags, print_tag = 0, 0, 0
reg_y = None
reg_x = None
y_predict = None
r2, new_r2 = None, None

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


class My_Nddata(Ui_dialog, QDialog):
    def __init__(self):
        super().__init__()

        self.child_Win = None

        self.setupUi(self)

        self.show()

        self.fileButton.clicked.connect(
            self.browsefile
        )
        self.colorButton.clicked.connect(
            self.showdialog
        )
        self.pushButton.clicked.connect(
            self.toplot
        )
        self.print_Button.clicked.connect(
            self.print_data
        )
        self.fileButton_1.clicked.connect(
            self.browsefile_1
        )
        self.fileButton_2.clicked.connect(
            self.browsefile_2
        )
        self.ld1data_Button.clicked.connect(
            self.ld1data
        )
        self.ld2data_Button.clicked.connect(
            self.ld2data
        )
        self.merge_Button.clicked.connect(
            self.merge
        )
        self.scatterxBox.valueChanged['int'].connect(
            self.renewbox
        )
        self.scatteryBox.valueChanged['int'].connect(
            self.renewbox
        )
        self.lddata_Button.clicked.connect(
            self.lddata
        )
        self.plotButton.clicked.connect(
            self.tosubplot
        )
        self.showButton.clicked.connect(
            self.subplots_show
        )
        self.subplotsrows.valueChanged['int'].connect(
            self.sub
        )
        self.subplotscolums.valueChanged['int'].connect(
            self.sub
        )
        self.regression.clicked.connect(
            self.lreg
        )
        self.regression_muti.clicked.connect(
            self.muti
        )
        self.regression_ln.clicked.connect(
            self.ln
        )
        self.regression_ex.clicked.connect(
            self.ex
        )
        self.regression_p.clicked.connect(
            self.power
        )
        self.fileButton_red.clicked.connect(
            self.browsred
        )
        self.fileButton_nir.clicked.connect(
            self.browsnir
        )
        self.ndvi_mean.clicked.connect(
            self.main
        )
        self.to_csv.clicked.connect(
            self.to_csv_fuc
        )
        self.lon_latshow.clicked.connect(
            self.lon_lat_fuc
        )
        self.residual_analysis.clicked.connect(
            self.residual_analysis_fuc
        )
        self.conclusion_Button.clicked.connect(
            self.conclusion_fuc
        )
        self.auto_reg.clicked.connect(
            self.auto_r
        )
        self.batch.clicked.connect(
            self.bat_bro
        )
        self.tabWidget_2.currentChanged.connect(
            self.change_tab
        )
        self.year_ndvi.clicked.connect(
            self.bat_years
        )
        self.clear_ndvi_arr.clicked.connect(
            self.clear_ndvi_arr_fuc
        )

    def change_tab(self):
        if self.tabWidget_2.currentIndex() == 1:
            self.label_59.setStyleSheet("color: red")
        if self.tabWidget_2.currentIndex() == 0:
            self.label_59.setStyleSheet("color: black")

    def toplot(self):
        global data
        if data.empty:
            QMessageBox.warning(self, "警告", "你还没有导入数据，请重新导入数据")
            return 0
        if self.tabWidget.currentIndex() == 0:
            self.rplot()
        if self.tabWidget.currentIndex() == 1:
            self.rbar()
        if self.tabWidget.currentIndex() == 2:
            self.rhist()
        if self.tabWidget.currentIndex() == 3:
            self.rscatter()
        if self.tabWidget.currentIndex() == 4:
            self.rpie()
        if self.tabWidget.currentIndex() == 5:
            self.rbox()

    def sub(self):
        global axes, fig, sub_rows, sub_colums
        sub_rows = self.subplotsrows.value()
        sub_colums = self.subplotscolums.value()
        fig, axes = plt.subplots(sub_rows, sub_colums)

    def save_ndvi_as_tif(self, output_file, ndvi_array, dataset):
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(output_file, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)

        out_dataset.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset.SetProjection(dataset.GetProjection())

        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(ndvi_array)
        out_band.FlushCache()
        out_dataset = None

    def transform_to_wgs84(self, src_epsg, dataset):
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(src_epsg)

        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromEPSG(4326)

        transform = osr.CoordinateTransformation(src_srs, tgt_srs)
        rever_transform = osr.CoordinateTransformation(tgt_srs, src_srs)
        return transform, rever_transform

    def get_lat_lon_bounds(self, dataset, transform):
        gt = dataset.GetGeoTransform()
        x_size = dataset.RasterXSize
        y_size = dataset.RasterYSize

        x_min, y_max = gt[0], gt[3]
        x_max, y_min = gt[0] + x_size * gt[1], gt[3] + y_size * gt[5]
        (ul_lon, ul_lat, _) = transform.TransformPoint(x_min, y_max)
        (lr_lon, lr_lat, _) = transform.TransformPoint(x_max, y_min)

        return ul_lon, ul_lat, lr_lon, lr_lat

    def browsred(self):
        file_name_red = QFileDialog.getOpenFileName(self, '选择单个文件', filter='*.tif')
        self.line_red.setText(file_name_red[0])

    def browsnir(self):
        file_name_nir = QFileDialog.getOpenFileName(self, '选择单个文件', filter='*.tif')
        self.line_nir.setText(file_name_nir[0])

    def calculate_ndvi(self, red_band, nir_band):
        red = red_band.astype(np.float64)
        nir = nir_band.astype(np.float64)
        numerator = nir - red
        denominator = nir + red
        denominator[denominator == 0] = 0.00001  # 将分母中为0的元素替换为0.0001
        ndvi = numerator / denominator
        ndvi = ndvi.astype(np.float64)
        return ndvi

    def main(self):
        global ndvi_arr, tag
        red_file = self.line_red.text()
        nir_file = self.line_nir.text()
        # red_file = r"D:\hp\DATA\LC08_L2SP_123032_20210619_20210628_02_T1_SR_B4.TIF"
        # nir_file = r"D:\hp\DATA\LC08_L2SP_123032_20210619_20210628_02_T1_SR_B5.TIF"
        red_dataset = gdal.Open(red_file)
        nir_dataset = gdal.Open(nir_file)

        red_band = red_dataset.GetRasterBand(1).ReadAsArray()
        nir_band = nir_dataset.GetRasterBand(1).ReadAsArray()

        ndvi = self.calculate_ndvi(red_band, nir_band)
        self.save_ndvi_as_tif("ndvi_output.tif", ndvi, red_dataset)

        srs = osr.SpatialReference(wkt=red_dataset.GetProjection())
        src_epsg = int(srs.GetAttrValue('AUTHORITY', 1))
        transform, rever_transform = self.transform_to_wgs84(src_epsg, red_dataset)

        ul_lon, lr_lat, lr_lon, ul_lat = self.get_lat_lon_bounds(red_dataset, transform)
        print(f"经纬度范围: ({lr_lon}, {lr_lat}) - ({ul_lon}, {ul_lat}) ")

        lat_max = float(self.line_au.text()) if self.line_au.text() else ul_lat
        lat_min = float(self.line_al.text()) if self.line_al.text() else lr_lat
        lon_min = float(self.line_ol.text()) if self.line_ol.text() else lr_lon
        lon_max = float(self.line_ou.text()) if self.line_ou.text() else ul_lon

        # 打开遥感数据文件
        raster = gdal.Open(red_file)
        # 获取地理变换信息
        gt = raster.GetGeoTransform()
        cols = raster.RasterXSize
        rows = raster.RasterYSize
        # 将经纬度转换为像素坐标范围
        x_min = gt[0]
        y_max = gt[3]
        x_max = gt[0] + cols * gt[1] + rows * gt[2]
        y_min = gt[3] + cols * gt[4] + rows * gt[5]

        px_min = int((x_min - gt[0]) / gt[1])  # 左上角的x坐标
        px_max = int((x_max - gt[0]) / gt[1])  # 右下角的x坐标
        py_max = int((y_min - gt[3]) / gt[5])  # 左上角的y坐标
        py_min = int((y_max - gt[3]) / gt[5])  # 右下角的y坐标

        oxmin = round((lat_min - lr_lat) / (ul_lat - lr_lat) * (px_max - px_min))
        oxmax = round((lat_max - lr_lat) / (ul_lat - lr_lat) * (px_max - px_min))
        oymin = round((lon_min - lr_lon) / (ul_lon - lr_lon) * (py_max - py_min))
        oymax = round((lon_max - lr_lon) / (ul_lon - lr_lon) * (py_max - py_min))

        # 选择感兴趣的像素并计算平均值
        ndvi_roi = ndvi[oxmin:oxmax, oymin:oymax]
        for i in range(ndvi_roi.shape[0]):
            for j in range(ndvi_roi.shape[1]):
                if ndvi_roi[i][j] == 0 and red_band[i + oxmin][j + oymin] == 0:
                    ndvi_roi[i][j] = np.nan
        ndvi_roi = ndvi_roi[~np.isnan(ndvi_roi)]
        if ndvi_roi.size == 0:
            QMessageBox.warning(self, "警告", "所选区域无有效像素！")
            return 0
        mean_ndvi = np.mean(ndvi_roi)
        ndvi_arr = pd.concat(
            [ndvi_arr, pd.DataFrame({'time': [self.ndvi_row_label.text()], 'ndvi_mean': [round(mean_ndvi, 4)]})],
            ignore_index=True)

        tag += 1
        self.ndvi_num.setText(f'{tag}')
        # self.ndvi_row_label_show.setText(ndvi_arr.iloc[:, 0].to_string(index=False).replace("\n", ","))
        # 将 DataFrame 转换成字符串，并用逗号分隔
        show_text = ndvi_arr.iloc[:, 0].to_string(index=False).replace("\n", ",")
        # 将字符串按照每 40 个字符分割成多行
        lines = [show_text[i:i + 40] for i in range(0, len(show_text), 40)]
        # 使用换行符将多行拼接成一个字符串
        formatted_text = "\n".join(lines)
        # 将处理后的字符串设置为 label 的文本
        self.ndvi_row_label_show.setText(formatted_text)
        self.ndvi_meano.setText(f"指定经纬度范围内的NDVI平均值:\n {mean_ndvi}")

    def lon_lat_fuc(self):
        try:
            if not self.line_red.text() and not self.line_nir.text():
                QMessageBox.warning(self, "提示", "没有任何遥感数据")
                return 0
            transform_file = self.line_red.text() if self.line_red.text() else self.line_nir.text()
            red_dataset = gdal.Open(transform_file)
            srs = osr.SpatialReference(wkt=red_dataset.GetProjection())
            src_epsg = int(srs.GetAttrValue('AUTHORITY', 1))
            transform, rever_transform = self.transform_to_wgs84(src_epsg, red_dataset)
            ul_lon, lr_lat, lr_lon, ul_lat = self.get_lat_lon_bounds(red_dataset, transform)
            self.lon_lat.setText(f"经纬度范围: \n({lr_lon}, {lr_lat}) - ({ul_lon}, {ul_lat}) ")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"出现异常：{e}")
            return 0

    def to_csv_fuc(self):
        global ndvi_arr
        try:
            ndvi_arr = ndvi_arr.sort_values(by='time')
            ndvi_arr.to_csv("ndvi_mean.csv", index=False, encoding="gbk")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"出现异常：{e}")

    def tosubplot(self):
        try:
            global x_col, y_col
            global xlabel, ylabel
            global kind, axes
            if not axes:
                QMessageBox.warning(self, "提示", "如果要绘制单图形，请选择单图形绘制")
                return 0
            if self.tabWidget.currentIndex() == 3:
                xlabel = self.scatterxlabelEdit.text()
                ylabel = self.scatterylabelEdit.text()
                x_col = (self.scatterxBox.value()) - 1
                y_col = self.scatteryBox.value() - 1
                kind = 'scatter'
            if self.tabWidget.currentIndex() == 0:
                xlabel = self.plotxlabelEdit.text()
                ylabel = self.plotylabelEdit.text()
                x_col = None
                y_col = None
                kind = 'line'
            if self.tabWidget.currentIndex() == 1:
                xlabel = self.barxlabelEdit.text()
                ylabel = self.barylabelEdit.text()
                x_col = None
                y_col = None
                kind = 'barh' if self.barhButton.isChecked() else 'bar'
            if self.tabWidget.currentIndex() == 2:
                xlabel = None
                ylabel = None
                x_col = None
                y_col = None
                kind = 'hist'
            if self.tabWidget.currentIndex() == 4:
                xlabel = None
                ylabel = self.pieylabelEdit.text()
                x_col = None
                y_col = None
                kind = 'pie'
            if self.tabWidget.currentIndex() == 5:
                xlabel = None
                ylabel = self.boxylabelEdit.text()
                x_col = None
                y_col = None
                kind = 'box'
            self.subplots()
        except Exception as e:
            QMessageBox.warning(self, "提示", f"出现异常：{e}")

    def subplots(self):
        # 设置参数
        plt.rcParams['figure.figsize'] = self.sizeEdit.text() if self.sizeEdit.text() else (5, 3)
        plt.rcParams['figure.dpi'] = self.dpiEdit.text() if self.dpiEdit.text() else 100
        global data, axes
        style = None
        if data.empty:
            QMessageBox.warning(self, "警告", "你还没有导入数据，请重新导入数据")
            return 0
        # 颜色条选择
        color = self.col_label.text()
        if not self.plotlinestyleEdit.text() and self.tabWidget.currentIndex() == 0:
            style = self.plotlinestyleEdit.text()
        cmap = None
        if self.cmapbox.currentIndex() != 0:
            if self.tabWidget.currentIndex() != 3 and self.tabWidget.currentIndex() != 1:
                QMessageBox.warning(self, "警告", "只有柱状图和散点图有颜色条")
                return 0
            try:
                color = list(data[self.cmaptext.text()])
            except KeyError:
                QMessageBox.warning(self, "警告", "不能把您选取的列作为颜色条依据，请打印数据检查列名或取消颜色条选择")
                return 0
            if self.cmapbox.currentIndex() == 1:
                cmap = 'binary'
            elif self.cmapbox.currentIndex() == 2:
                cmap = 'PiYG'
            elif self.cmapbox.currentIndex() == 3:
                cmap = 'Blues'
            elif self.cmapbox.currentIndex() == 4:
                cmap = 'viridis'
        # 子图设置
        current_rows = self.axrows.value() - 1
        current_colums = self.axcolums.value() - 1
        if sub_rows != 1 and sub_colums == 1:
            QMessageBox.warning(self, "警告", "不支持行大于1列等于1的分割方式，请把行列互换")
            return 0
        if sub_rows == 1:
            if sub_colums == 1:
                current_axes = axes
            else:
                current_axes = axes[current_colums]
        else:
            current_axes = axes[current_rows][current_colums]
        # 绘图
        global x_col, y_col
        global xlabel, ylabel
        global kind
        data.plot(x=x_col, y=y_col, kind=kind, title=self.titleEdit.text(), style=style, c=color,
                  xlabel=xlabel, cmap=cmap, subplots=True, ax=current_axes, ylabel=ylabel, rot=int(self.rotEdit.text()))

        # 部分参数还原
        style = None

        # 获取x，y轴的列名
        if self.tabWidget.currentIndex() == 3:
            names = list(data.columns)
            self.x_name.setText(names[x_col])
            self.y_name.setText(names[y_col])

    def subplots_show(self):
        # 保存绘制结果并展示
        savepath = QFileDialog.getSaveFileName(self, '选择保存路径', filter='*.png;*.jpg')
        filename = savepath[0].split('/')[-1]
        self.savepathEdit.clear()
        self.savepathEdit.setText(savepath[0])
        plt.tight_layout()
        plt.savefig(savepath[0])
        self.lbl.resize(400, 240)
        self.lbl.setScaledContents(True)
        self.lbl.setPixmap(QPixmap(filename))

    def renewbox(self):
        global data, x_col, y_col
        if data.empty:
            pass
        else:
            x_col = (self.scatterxBox.value()) - 1
            y_col = self.scatteryBox.value() - 1
            names = list(data.columns)
            self.x_name.setText(names[x_col])
            self.y_name.setText(names[y_col])

    def browsefile(self):
        file_name = QFileDialog.getOpenFileName(self, '选择单个文件', filter='*.txt;*.csv')
        self.pathEdit.setText(file_name[0])

    def browsefile_1(self):
        file_name = QFileDialog.getOpenFileName(self, '选择单个文件', filter='*.txt;*.csv')
        self.pathEdit_1.setText(file_name[0])

    def browsefile_2(self):
        file_name = QFileDialog.getOpenFileName(self, '选择单个文件', filter='*.txt;*.csv')
        self.pathEdit_2.setText(file_name[0])

    def lddata(self):
        global data, print_tag
        if not self.colsEdit.text():
            QMessageBox.warning(self, "警告", "请先输入使用的是哪几列数据")
            return 0
        if self.skipBox.value() == -1:
            QMessageBox.warning(self, "警告", "请先输入需要跳过的行数")
            return 0
        data = self.load(path=self.pathEdit.text())
        print_tag = 0

    def ld1data(self):
        global data_1, print_tag
        if not self.colsEdit.text():
            QMessageBox.warning(self, "警告", "请先输入使用的是哪几列数据")
            return 0
        if self.skipBox.value() == -1:
            QMessageBox.warning(self, "警告", "请先输入需要跳过的行数")
            return 0
        data_1 = self.load(path=self.pathEdit_1.text(), source=1)
        print_tag = 1

    def ld2data(self):
        global data_2, print_tag
        if not self.colsEdit.text():
            QMessageBox.warning(self, "警告", "请先输入使用的是哪几列数据")
            return 0
        if self.skipBox.value() == -1:
            QMessageBox.warning(self, "警告", "请先输入需要跳过的行数")
            return 0
        data_2 = self.load(path=self.pathEdit_2.text(), source=2)
        print_tag = 2

    def load(self, path, source=0):
        skiprows = self.skipBox.value()
        clos = list((np.array(list(self.colsEdit.text()), dtype=int)) - 1)
        bol = 0
        # 判断数据中分隔符类型
        delim = self.delimEdit.text()
        if source == 0:
            bol = bool(self.pathEdit.text().endswith('.csv'))
        elif source == 1:
            bol = bool(self.pathEdit_1.text().endswith('.csv'))
        elif source == 2:
            bol = bool(self.pathEdit_2.text().endswith('.csv'))
        if not delim:
            if bol:
                delimiter = ','
                delim_whitespace = False
            else:
                delim_whitespace = True
                delimiter = None
        else:
            delim_whitespace = False
            delimiter = delim
        # 自定义的数据类型
        dtypedict = {0: 'float', 1: 'int', 2: 'str', 3: 'object'}
        dtyp_index = self.dtypeBox.currentIndex()
        dt = dtypedict[dtyp_index]
        if self.to_dtypeBox.value():
            dtype_list = [(self.to_dtypeBox.value(), dt)]
            dtype = dict(dtype_list)
        else:
            dtype = None
        # 确定列标签
        if not self.indexnumEdit.value():
            if not self.indexEdit.text():
                index = None
            else:
                index = list(map(int, self.indexEdit.text().split(',')))
        else:
            index = list(range(self.indexnumEdit.value()))
        try:
            if not self.headerEdit.value():
                header = 'infer'
                if not self.namesEdit.text():
                    names = list(range(len(clos)))
                else:
                    names = self.namesEdit.text().split(',')
                dataframe = pd.read_csv(path, usecols=clos, delimiter=delimiter, skiprows=skiprows, encoding='utf-8',
                                        delim_whitespace=delim_whitespace, names=names, header=header, index_col=index,
                                        dtype=dtype)
            else:
                header = self.headerEdit.value() - 1
                dataframe = pd.read_csv(path, usecols=clos, delimiter=delimiter, skiprows=skiprows, encoding='utf-8',
                                        delim_whitespace=delim_whitespace, header=header, index_col=index, dtype=dtype)
        except pd.errors.EmptyDataError:
            QMessageBox.warning(self, "导入失败", "重新检查您输入的必要参数是否匹配您的文件")
            return 0
        if self.TButton.isChecked():
            dataframe = dataframe.T
        else:
            pass
        return dataframe

    def merge(self):
        global data, data_1, data_2, print_tag
        how = None
        left_index = False
        right_index = False
        if self.lrindex.isChecked():
            left_index = True
            right_index = True
        if self.how_left.isChecked():
            how = 'left'
        elif self.how_right.isChecked():
            how = 'right'
        elif self.how_in.isChecked():
            how = 'inner'
        elif self.how_out.isChecked():
            how = 'outer'
        elif self.how_cross.isChecked():
            how = 'cross'
        if self.merge_onEdit.text():
            on = self.merge_onEdit.text().split(',')
        else:
            on = None
        data = pd.merge(data_1, data_2, how=how, on=on, left_index=left_index, right_index=right_index)
        print_tag = 0
        print(data)

    def print_data(self):
        if print_tag == 0:
            if data.empty:
                QMessageBox.warning(self, "警告", "您还没有导入数据")
                return 0
            QMessageBox.about(self, "展示数据", str(data))
        elif print_tag == 1:
            if data_1.empty:
                QMessageBox.warning(self, "警告", "您还没有导入数据")
                return 0
            QMessageBox.about(self, "展示数据", str(data_1))
        elif print_tag == 2:
            if data_2.empty:
                QMessageBox.warning(self, "警告", "您还没有导入数据")
                return 0
            QMessageBox.about(self, "展示数据", str(data_2))

    def showdialog(self):
        selc = QColorDialog.getColor()
        self.col_label.setText(selc.name())
        if selc.isValid():
            self.colorwidget.setStyleSheet('QWidget {background-color:%s}' % selc.name())

    def lreg(self):
        global x_col, y_col, reg_y, y_predict, reg_x, r2
        temp_x = data.iloc[:, x_col]
        temp_y = data.iloc[:, y_col]
        reg_x = np.array(temp_x).reshape(-1, 1)
        reg_y = np.array(temp_y)
        lr1 = LinearRegression()
        lr1.fit(reg_x, reg_y)
        y_predict = lr1.predict(reg_x)
        r2 = r2_score(reg_y, y_predict)
        if if_show == 0:
            plt.scatter(reg_y, y_predict)
            plt.xlabel("实际值")
            plt.ylabel("预测值")
            if not os.path.exists('analysis results'):
                os.makedirs('analysis results')
            plt.savefig('analysis results/lreg.png')
            plt.show()
        self.regression_o.setText("r2 = %f\n线性回归方程为y=%fx+%f" % (r2, lr1.coef_, lr1.intercept_))

    def muti(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        global x_col, y_col, reg_y, y_predict, reg_x, r2
        temp_x = data.iloc[:, x_col]
        temp_y = data.iloc[:, y_col]
        reg_x = np.array(temp_x).reshape(-1, 1)
        reg_y = np.array(temp_y)
        poly = PolynomialFeatures(degree=2)
        poly.fit(reg_x)
        reg_x_2 = poly.transform(reg_x).reshape(-1, 3)
        estimator_2 = LinearRegression()
        estimator_2.fit(reg_x_2, reg_y)
        y_predict = estimator_2.predict(reg_x_2)
        r2 = r2_score(reg_y, y_predict)
        if if_show == 0:
            plt.scatter(reg_y, y_predict)
            plt.xlabel("实际值")
            plt.ylabel("预测值")
            if not os.path.exists('analysis results'):
                os.makedirs('analysis results')
            plt.savefig('analysis results/muti.png')
            plt.show()
        self.regression_om.setText("r2 = %f\n多项式回归方程为y=%fx**2+%fx+%f" % (r2, estimator_2.coef_[2],
                                                                         estimator_2.coef_[1], estimator_2.intercept_))

    def ln(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        global x_col, y_col, reg_y, y_predict, reg_x, r2
        temp_x = data.iloc[:, x_col]
        temp_y = data.iloc[:, y_col]
        reg_x = np.array(temp_x)
        reg_y = np.array(temp_y)

        def func(x, a, c):
            return a * np.log(x) + c

        # reg_x = np.linspace(0.01, 4, 50)
        # y = func(reg_x, 2.5, 1)
        # rng = np.random.default_rng()
        # y_noise = 0.2 * rng.normal(size=reg_x.size)
        # reg_y = y

        popt, pcov = curve_fit(func, reg_x, reg_y)

        y_predict = func(reg_x, *popt)
        r2 = r2_score(reg_y, y_predict)
        if if_show == 0:
            plt.scatter(reg_y, y_predict)
            plt.xlabel("实际值")
            plt.ylabel("预测值")
            if not os.path.exists('analysis results'):
                os.makedirs('analysis results')
            plt.savefig('analysis results/ln.png')
            plt.show()
        self.regression_oln.setText("r2 = %f\n对数回归方程为y=%f * ln(x) + %f" % (r2, popt[0], popt[1]))

    def ex(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        global x_col, y_col, reg_y, y_predict, reg_x, r2
        temp_x = data.iloc[:, x_col]
        temp_y = data.iloc[:, y_col]
        reg_x = np.array(temp_x)
        reg_y = np.array(temp_y)

        def func(x, a, b):
            return a * np.exp(b * x)

        # reg_x = np.linspace(0, 4, 50)
        # y = func(reg_x, 2.5, 1.3, 0.5)
        # rng = np.random.default_rng()
        # y_noise = 0.2 * rng.normal(size=reg_x.size)
        # reg_y = y + y_noise
        try:
            popt, pcov = curve_fit(func, reg_x, reg_y)
        except RuntimeError:
            QMessageBox.warning(self, "警告", "无法拟合")
            return 0

        y_predict = func(reg_x, *popt)
        r2 = r2_score(reg_y, y_predict)
        if if_show == 0:
            plt.scatter(reg_y, y_predict)
            plt.xlabel("实际值")
            plt.ylabel("预测值")
            if not os.path.exists('analysis results'):
                os.makedirs('analysis results')
            plt.savefig('analysis results/oex.png')
            plt.show()
        self.regression_oex.setText("r2 = %f\n指数回归方程为y=%f * exp(%f * x)" % (r2, popt[0], popt[1]))

    def power(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        global x_col, y_col, reg_y, y_predict, reg_x, r2
        temp_x = data.iloc[:, x_col]
        temp_y = data.iloc[:, y_col]
        reg_x = np.array(temp_x)
        reg_y = np.array(temp_y)

        def func(x, a, b):
            return a * np.power(x, b)

        try:
            popt, pcov = curve_fit(func, reg_x, reg_y)
        except RuntimeError:
            QMessageBox.warning(self, "警告", "无法拟合")
            return 0

        y_predict = func(reg_x, *popt)
        r2 = r2_score(reg_y, y_predict)
        if if_show == 0:
            plt.scatter(reg_y, y_predict)
            plt.xlabel("实际值")
            plt.ylabel("预测值")
            if not os.path.exists('analysis results'):
                os.makedirs('analysis results')
            plt.savefig('analysis results/power.png')
            plt.show()
        self.regression_op.setText("r2 = %f\n幂函数回归方程为y=%f * x^%f" % (r2, popt[0], popt[1]))

    def residual_analysis_fuc(self):
        global reg_x, reg_y, y_predict, new_r2, r2, tags
        if reg_x is None:
            QMessageBox.warning(self, "警告", "请先进行回归分析")
            return 0
        residual = reg_y - y_predict
        mean_residual = np.mean(residual)
        std_residual = np.std(residual)
        standardized_residual = (residual - mean_residual) / std_residual
        for i in standardized_residual:
            if i < -2 or i > 2:
                tags = 1
                break
        new_data = pd.concat([pd.DataFrame(reg_x), pd.DataFrame(reg_y)], axis=1)
        new_data.columns = ['x', 'y']
        new_data['residual'] = residual
        new_data['standardized_residual'] = standardized_residual
        abnormal_data = new_data[(new_data['standardized_residual'] < -2) | (new_data['standardized_residual'] > 2)]
        new_data = new_data[(new_data['standardized_residual'] > -2) & (new_data['standardized_residual'] < 2)]
        new_reg_x = new_data['x'].values.reshape(-1, 1)
        new_reg_y = new_data['y'].values.reshape(-1, 1)
        if tags == 1:
            if if_show == 0:
                QMessageBox.warning(self, "警告", "存在异常实验点，已剔除,如下异常值项\n（忽略第一个编号）\n从左到右其次为%s" % abnormal_data)
            if self.tabWidget_regression.currentIndex() == 0:
                lr1 = LinearRegression()
                lr1.fit(new_reg_x, new_reg_y)
                new_y_predict = lr1.predict(new_reg_x)
                new_r2 = r2_score(new_reg_y, new_y_predict)
                self.regression_R2.setText("%f" % new_r2)
                self.regression_o.setText("r2 = %f\n线性回归方程为y=%fx+%f" % (new_r2, lr1.coef_, lr1.intercept_))
            elif self.tabWidget_regression.currentIndex() == 1:
                poly = PolynomialFeatures(degree=2)
                poly.fit(new_reg_x)
                reg_x_2 = poly.transform(new_reg_x).reshape(-1, 3)
                estimator_2 = LinearRegression()
                estimator_2.fit(reg_x_2, new_reg_y)
                new_y_predict = estimator_2.predict(reg_x_2)
                new_r2 = r2_score(new_reg_y, new_y_predict)
                self.regression_muti_R2.setText("%f" % new_r2)
                self.regression_om.setText("r2 = %f\n多项式回归方程为y=%fx**2+%fx+%f" % (new_r2, estimator_2.coef_[0, 2],
                                                                                 estimator_2.coef_[0, 1],
                                                                                 estimator_2.intercept_))
            elif self.tabWidget_regression.currentIndex() == 2:
                func = lambda x, a, b: a * np.log(x) + b
                new_popt, new_pcov = curve_fit(func, new_reg_x.ravel(), new_reg_y.ravel())
                new_y_predict = func(new_reg_x, *new_popt)
                new_r2 = r2_score(new_reg_y, new_y_predict)
                self.regression_ln_R2.setText("%f" % new_r2)
                self.regression_oln.setText("r2 = %f\n对数函数回归方程为y=%f * ln(x) + %f" % (new_r2, new_popt[0], new_popt[1]))

            elif self.tabWidget_regression.currentIndex() == 3:
                func = lambda x, a, b: a * np.exp(b * x)
                new_popt, new_pcov = curve_fit(func, new_reg_x.ravel(), new_reg_y.ravel())
                new_y_predict = func(new_reg_x, *new_popt)
                new_r2 = r2_score(new_reg_y, new_y_predict)
                self.regression_ex_R2.setText("%f" % new_r2)
                self.regression_oex.setText("r2 = %f\n指数函数回归方程为y=%f * e^(%f * x)" % (new_r2, new_popt[0], new_popt[1]))

            elif self.tabWidget_regression.currentIndex() == 4:
                func = lambda x, a, b: a * np.power(x, b)
                new_popt, new_pcov = curve_fit(func, new_reg_x.ravel(), new_reg_y.ravel())
                new_y_predict = func(new_reg_x, *new_popt)
                new_r2 = r2_score(new_reg_y, new_y_predict)
                self.regression_p_R2.setText("%f" % new_r2)
                self.regression_op.setText(
                    "r2 = %f\n幂函数回归方程为y=%f * x^%f" % (new_r2, new_popt[0], new_popt[1]))
        else:
            new_y_predict = pd.DataFrame(y_predict)
            if self.tabWidget_regression.currentIndex() == 0:
                self.regression_R2.setText("%f" % r2)
            if self.tabWidget_regression.currentIndex() == 1:
                self.regression_muti_R2.setText("%f" % r2)
            if self.tabWidget_regression.currentIndex() == 2:
                self.regression_ln_R2.setText("%f" % r2)
            if self.tabWidget_regression.currentIndex() == 3:
                self.regression_ex_R2.setText("%f" % r2)
            if self.tabWidget_regression.currentIndex() == 4:
                self.regression_p_R2.setText("%f" % r2)
            if if_show == 0:
                QMessageBox.information(self, "提示", "不存在异常实验点")
        tags = 0  # 重置异常点标志位
        residual = new_reg_y - new_y_predict
        mean_residual = np.mean(residual)
        std_residual = np.std(residual)
        standardized_residual = (residual - mean_residual) / std_residual
        if if_show == 0:
            plt.scatter(new_y_predict, standardized_residual)
            plt.xlabel("去除异常点后的预测值")
            plt.ylabel("标准化残差值")
            plt.title("残差分析")
            plt.tight_layout()
            if not os.path.exists('analysis results'):
                os.makedirs('analysis results')
            plt.savefig('analysis results/residual_analysis.png')
            plt.show()

    def conclusion_fuc(self):
        global r2
        good_list = []
        r2_dict = {"线性回归": float(self.regression_R2.text()), "多项式回归": float(self.regression_muti_R2.text()),
                   "对数函数回归": float(self.regression_ln_R2.text()), "指数函数回归": float(self.regression_ex_R2.text()),
                   "幂函数回归": float(self.regression_p_R2.text())}
        max_r2 = max(r2_dict.values())
        min_r2 = min(r2_dict.values())
        for key, value in r2_dict.items():
            if value == max_r2:
                good_list.append(key)
        if float(self.regression_R2.text()) == max_r2:
            self.regression_R2.setStyleSheet("color: red")
        elif float(self.regression_R2.text()) == min_r2:
            self.regression_R2.setStyleSheet("color: blue")
        if float(self.regression_muti_R2.text()) == max_r2:
            self.regression_muti_R2.setStyleSheet("color: red")
        elif float(self.regression_muti_R2.text()) == min_r2:
            self.regression_muti_R2.setStyleSheet("color: blue")
        if float(self.regression_ln_R2.text()) == max_r2:
            self.regression_ln_R2.setStyleSheet("color: red")
        elif float(self.regression_ln_R2.text()) == min_r2:
            self.regression_ln_R2.setStyleSheet("color: blue")
        if float(self.regression_ex_R2.text()) == max_r2:
            self.regression_ex_R2.setStyleSheet("color: red")
        elif float(self.regression_ex_R2.text()) == min_r2:
            self.regression_ex_R2.setStyleSheet("color: blue")
        if float(self.regression_p_R2.text()) == max_r2:
            self.regression_p_R2.setStyleSheet("color: red")
        elif float(self.regression_p_R2.text()) == min_r2:
            self.regression_p_R2.setStyleSheet("color: blue")
        if good_list:
            self.conclusion.setText("%s\n的决定系数最接近1，故该模型对数据拟合较好" % good_list)

    def rbox(self):
        # 设置参数
        plt.rcParams['figure.figsize'] = self.sizeEdit.text() if self.sizeEdit.text() else (5, 3)
        plt.rcParams['figure.dpi'] = self.dpiEdit.text() if self.dpiEdit.text() else 100
        global data
        # 绘图
        data.plot(kind='box', title=self.titleEdit.text(),
                  ylabel=self.boxylabelEdit.text(), rot=int(self.rotEdit.text()))
        # 保存绘制结果并展示
        savepath = QFileDialog.getSaveFileName(self, '选择保存路径', filter='*.png;*.jpg')
        filename = savepath[0].split('/')[-1]
        self.savepathEdit.clear()
        self.savepathEdit.setText(savepath[0])
        plt.tight_layout()
        plt.savefig(savepath[0])
        self.lbl.setPixmap(QPixmap(filename))
        self.lbl.resize(400, 240)
        self.lbl.setScaledContents(True)
        plt.show()

    def rpie(self):
        # 设置参数
        plt.rcParams['figure.figsize'] = self.sizeEdit.text() if self.sizeEdit.text() else (5, 3)
        plt.rcParams['figure.dpi'] = self.dpiEdit.text() if self.dpiEdit.text() else 100
        global data
        # 绘图
        data.plot(kind='pie', title=self.titleEdit.text(),
                  ylabel=self.pieylabelEdit.text(), rot=int(self.rotEdit.text()))
        # 保存绘制结果并展示
        savepath = QFileDialog.getSaveFileName(self, '选择保存路径', filter='*.png;*.jpg')
        filename = savepath[0].split('/')[-1]
        self.savepathEdit.clear()
        self.savepathEdit.setText(savepath[0])
        plt.tight_layout()
        plt.savefig(savepath[0])
        self.lbl.setPixmap(QPixmap(filename))
        self.lbl.resize(400, 240)
        self.lbl.setScaledContents(True)
        plt.show()

    def rscatter(self):
        # 设置参数
        plt.rcParams['figure.figsize'] = self.sizeEdit.text() if self.sizeEdit.text() else (5, 3)
        plt.rcParams['figure.dpi'] = self.dpiEdit.text() if self.dpiEdit.text() else 100
        global data
        # 颜色条选择
        color = self.col_label.text()
        cmap = None
        if self.cmapbox.currentIndex() != 0:
            try:
                color = list(data[self.cmaptext.text()])
            except KeyError:
                QMessageBox.warning(self, "警告", "不能把您选取的列作为颜色条依据，请打印数据检查列名或取消颜色条选择")
                return 0
            if self.cmapbox.currentIndex() == 1:
                cmap = 'binary'
            elif self.cmapbox.currentIndex() == 2:
                cmap = 'PiYG'
            elif self.cmapbox.currentIndex() == 3:
                cmap = 'Blues'
            elif self.cmapbox.currentIndex() == 4:
                cmap = 'viridis'

        # 绘图
        global x_col, y_col
        x_col = (self.scatterxBox.value()) - 1
        y_col = self.scatteryBox.value() - 1
        data.plot(x=x_col, y=y_col, kind='scatter', title=self.titleEdit.text(),
                  xlabel=self.scatterxlabelEdit.text(), cmap=cmap,
                  c=color, ylabel=self.scatterylabelEdit.text(), rot=int(self.rotEdit.text()))

        # 获取x，y轴的列名
        names = list(data.columns)
        self.x_name.setText(names[x_col])
        self.y_name.setText(names[y_col])

        # 保存绘制结果并展示
        savepath = QFileDialog.getSaveFileName(self, '选择保存路径', filter='*.png;*.jpg')
        filename = savepath[0].split('/')[-1]
        self.savepathEdit.clear()
        self.savepathEdit.setText(savepath[0])
        plt.tight_layout()
        plt.savefig(savepath[0])
        self.lbl.setPixmap(QPixmap(savepath[0]))
        self.lbl.resize(400, 240)
        self.lbl.setScaledContents(True)
        plt.show()

    def rhist(self):
        # 设置参数
        plt.rcParams['figure.figsize'] = self.sizeEdit.text() if self.sizeEdit.text() else (5, 3)
        plt.rcParams['figure.dpi'] = self.dpiEdit.text() if self.dpiEdit.text() else 100

        # 直方图特殊参数
        color = self.col_label.text()

        # 绘图
        data.plot(kind='hist', title=self.titleEdit.text(), rot=int(self.rotEdit.text()),
                  bins=int(self.histbinsEdit.text()), color=color, alpha=float(self.histalphaEdit.text()))

        savepath = QFileDialog.getSaveFileName(self, '选择保存路径', filter='*.png;*.jpg')
        filename = savepath[0].split('/')[-1]
        self.savepathEdit.clear()
        self.savepathEdit.setText(savepath[0])
        plt.tight_layout()
        plt.savefig(savepath[0])
        self.lbl.setPixmap(QPixmap(filename))
        self.lbl.resize(400, 240)
        self.lbl.setScaledContents(True)
        plt.show()

    def rbar(self):
        # 设置参数
        plt.rcParams['figure.figsize'] = self.sizeEdit.text() if self.sizeEdit.text() else (5, 3)
        plt.rcParams['figure.dpi'] = self.dpiEdit.text() if self.dpiEdit.text() else 100
        global data
        isbarh = 'barh' if self.barhButton.isChecked() else 'bar'

        cmap = None
        if self.cmapbox.currentIndex() != 0:
            try:
                color = list(data[self.cmaptext.text()])
            except KeyError:
                QMessageBox.warning(self, "警告", "不能把您选取的列作为颜色条依据，请打印数据检查列名或取消颜色条选择")
                return 0
        else:
            color = None
            if self.cmapbox.currentIndex() == 1:
                cmap = 'binary'
            elif self.cmapbox.currentIndex() == 2:
                cmap = 'PiYG'
            elif self.cmapbox.currentIndex() == 3:
                cmap = 'Blues'
            elif self.cmapbox.currentIndex() == 4:
                cmap = 'viridis'

        # 绘图
        data.plot(kind=isbarh, title=self.titleEdit.text(), xlabel=self.barxlabelEdit.text(), style=color,
                  ylabel=self.barylabelEdit.text(), rot=int(self.rotEdit.text()), cmap=cmap)

        savepath = QFileDialog.getSaveFileName(self, '选择保存路径', filter='*.png;*.jpg')
        filename = savepath[0].split('/')[-1]
        self.savepathEdit.clear()
        self.savepathEdit.setText(savepath[0])
        plt.tight_layout()
        plt.savefig(savepath[0])
        self.lbl.setPixmap(QPixmap(filename))
        self.lbl.resize(400, 240)
        self.lbl.setScaledContents(True)
        plt.show()

    def rplot(self):
        # 设置参数
        plt.rcParams['figure.figsize'] = self.sizeEdit.text() if self.sizeEdit.text() else (5, 3)
        plt.rcParams['figure.dpi'] = self.dpiEdit.text() if self.dpiEdit.text() else 100
        global data
        # 绘图
        line_style = self.plotlinestyleEdit.text() if self.plotlinestyleEdit.text() else 'r*-'
        data.plot(title=self.titleEdit.text(), style=line_style, xlabel=self.plotxlabelEdit.text(),
                  ylabel=self.plotylabelEdit.text(), rot=int(self.rotEdit.text()))
        # 保存绘制结果并展示
        savepath = QFileDialog.getSaveFileName(self, '选择保存路径', filter='*.png;*.jpg')
        filename = savepath[0].split('/')[-1]
        self.savepathEdit.clear()
        self.savepathEdit.setText(savepath[0])
        plt.tight_layout()
        plt.savefig(savepath[0])
        self.lbl.setPixmap(QPixmap(filename))
        self.lbl.resize(400, 240)
        self.lbl.setScaledContents(True)
        plt.show()

    def auto_r(self):
        global if_show
        if_show = 1
        self.regression_resi()
        self.regression_muti_resi()
        self.regression_ln_resi()
        self.regression_ex_resi()
        self.regression_p_resi()
        self.conclusion_fuc()
        if_show = 0

    def regression_resi(self):
        self.lreg()
        self.tabWidget_regression.setCurrentIndex(0)
        self.residual_analysis_fuc()

    def regression_muti_resi(self):
        self.muti()
        self.tabWidget_regression.setCurrentIndex(1)
        self.residual_analysis_fuc()

    def regression_ln_resi(self):
        self.ln()
        self.tabWidget_regression.setCurrentIndex(2)
        self.residual_analysis_fuc()

    def regression_ex_resi(self):
        self.ex()
        self.tabWidget_regression.setCurrentIndex(3)
        self.residual_analysis_fuc()

    def regression_p_resi(self):
        self.power()
        self.tabWidget_regression.setCurrentIndex(4)
        self.residual_analysis_fuc()

    def bat(self):
        global ndvi_arr, tag
        red_file = ''
        nir_file = ''
        threads = []
        for root, dirs, files in os.walk(self.batch_data.text()):
            for file in files:
                if file.endswith("B4.TIF"):
                    red_file = os.path.join(root, file)
                if file.endswith("B5.TIF"):
                    nir_file = os.path.join(root, file)
                if red_file and nir_file:
                    pattern = r'\d{8}'
                    temp = re.search(pattern, file)
                    match = temp.group()[:6]
                    self.line_red.setText(red_file)
                    self.line_nir.setText(nir_file)
                    t = threading.Thread(target=self.main_bat, args=(match,))
                    threads.append(t)
                    t.start()
                    tread_num = self.Thread.value()
                    if len(threads) == tread_num:
                        for t in threads:
                            try:
                                t.join()
                            except MemoryError:
                                QMessageBox.warning(self, "警告", "内存或栈溢出，请减少线程数量并清除数据后重新进行批处理")
                                break
                        threads = []
                    break
            red_file = ''
            nir_file = ''
        for t in threads:
            t.join()
        self.ndvi_num.setText(f'{tag}')
        QMessageBox.information(self, "提示", "批处理完成,如果数据缺失，代表所选区域对应的遥感影像无数据")

    def main_bat(self, match=''):
        global ndvi_arr, tag
        red_file = self.line_red.text()
        nir_file = self.line_nir.text()
        red_dataset = gdal.Open(red_file)
        nir_dataset = gdal.Open(nir_file)

        red_band = red_dataset.GetRasterBand(1).ReadAsArray()
        nir_band = nir_dataset.GetRasterBand(1).ReadAsArray()

        ndvi = self.calculate_ndvi(red_band, nir_band)
        # self.save_ndvi_as_tif("ndvi_output.tif", ndvi, red_dataset)

        srs = osr.SpatialReference(wkt=red_dataset.GetProjection())
        src_epsg = int(srs.GetAttrValue('AUTHORITY', 1))
        transform, rever_transform = self.transform_to_wgs84(src_epsg, red_dataset)

        ul_lon, lr_lat, lr_lon, ul_lat = self.get_lat_lon_bounds(red_dataset, transform)
        print(f"经纬度范围: ({lr_lon}, {lr_lat}) - ({ul_lon}, {ul_lat}) ")

        lat_max = float(self.line_au.text()) if self.line_au.text() else ul_lat
        lat_min = float(self.line_al.text()) if self.line_al.text() else lr_lat
        lon_min = float(self.line_ol.text()) if self.line_ol.text() else lr_lon
        lon_max = float(self.line_ou.text()) if self.line_ou.text() else ul_lon

        # 打开遥感数据文件
        raster = gdal.Open(red_file)
        # 获取地理变换信息
        gt = raster.GetGeoTransform()
        cols = raster.RasterXSize
        rows = raster.RasterYSize
        # 将经纬度转换为像素坐标范围
        x_min = gt[0]
        y_max = gt[3]
        x_max = gt[0] + cols * gt[1] + rows * gt[2]
        y_min = gt[3] + cols * gt[4] + rows * gt[5]

        px_min = int((x_min - gt[0]) / gt[1])  # 左上角的x坐标
        px_max = int((x_max - gt[0]) / gt[1])  # 右下角的x坐标
        py_max = int((y_min - gt[3]) / gt[5])  # 左上角的y坐标
        py_min = int((y_max - gt[3]) / gt[5])  # 右下角的y坐标

        oxmin = round((lat_min - lr_lat) / (ul_lat - lr_lat) * (px_max - px_min))
        oxmax = round((lat_max - lr_lat) / (ul_lat - lr_lat) * (px_max - px_min))
        oymin = round((lon_min - lr_lon) / (ul_lon - lr_lon) * (py_max - py_min))
        oymax = round((lon_max - lr_lon) / (ul_lon - lr_lon) * (py_max - py_min))

        # 选择感兴趣的像素并计算平均值
        ndvi_roi = ndvi[oxmin:oxmax, oymin:oymax]
        for i in range(ndvi_roi.shape[0]):
            for j in range(ndvi_roi.shape[1]):
                if ndvi_roi[i][j] == 0 and red_band[i + oxmin][j + oymin] == 0:
                    ndvi_roi[i][j] = np.nan
        ndvi_roi = ndvi_roi[~np.isnan(ndvi_roi)]
        if ndvi_roi.size == 0:
            return 0
        mean_ndvi = np.mean(ndvi_roi)

        ndvi_arr = pd.concat(
            [ndvi_arr, pd.DataFrame({'time': [match], 'ndvi_mean': [round(mean_ndvi, 4)]})],
            ignore_index=True)
        tag += 1

    def bat_years(self):
        global ndvi_arr
        print(ndvi_arr)
        ndvi_arr['year'] = ndvi_arr['time'].str[:4]
        ndvi_arr = ndvi_arr.groupby('year').mean().reset_index()

    def bat_bro(self):
        file_name_red = QFileDialog.getExistingDirectory(self, '选择文件夹')
        self.batch_data.setText(file_name_red)
        self.bat()

    def clear_ndvi_arr_fuc(self):
        global ndvi_arr, tag
        ndvi_arr = pd.DataFrame()
        tag = 0
        self.ndvi_num.setText(f'{tag}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_nddata = My_Nddata()
    sys.exit(app.exec())
