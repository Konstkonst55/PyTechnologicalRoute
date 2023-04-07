# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\College\Other\PyProjects\PyTechnologicalRoute\ui\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class MainWindowUi(object):
    def setup_ui(self, main_window):
        main_window.setObjectName("main_window")
        main_window.resize(671, 668)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(main_window.sizePolicy().hasHeightForWidth())
        main_window.setSizePolicy(sizePolicy)
        self.central = QtWidgets.QWidget(main_window)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.central.sizePolicy().hasHeightForWidth())
        self.central.setSizePolicy(sizePolicy)
        self.central.setObjectName("central")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.central)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabs = QtWidgets.QTabWidget(self.central)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.tabs.setFont(font)
        self.tabs.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabs.setStyleSheet("\n"
                                "border: 0px solid black;\n"
                                "")
        self.tabs.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabs.setElideMode(QtCore.Qt.ElideNone)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setDocumentMode(False)
        self.tabs.setTabsClosable(False)
        self.tabs.setMovable(True)
        self.tabs.setTabBarAutoHide(True)
        self.tabs.setObjectName("tabs")
        self.tab_osn = QtWidgets.QWidget()
        self.tab_osn.setObjectName("tab_osn")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_osn)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.hl_top = QtWidgets.QHBoxLayout()
        self.hl_top.setObjectName("hl_top")
        self.b_learn_open = QtWidgets.QPushButton(self.tab_osn)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_learn_open.sizePolicy().hasHeightForWidth())
        self.b_learn_open.setSizePolicy(sizePolicy)
        self.b_learn_open.setMinimumSize(QtCore.QSize(0, 40))
        self.b_learn_open.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_learn_open.setFont(font)
        self.b_learn_open.setStyleSheet("QPushButton{\n"
                                        "border-radius: 5px;\n"
                                        "border: 1px solid #2B2A29;\n"
                                        "padding: 10px 10px 10px 10px;\n"
                                        "color: black;\n"
                                        "}\n"
                                        "\n"
                                        "QPushButton:hover{\n"
                                        "background: solid #625B71;\n"
                                        "color: white;\n"
                                        "}")
        self.b_learn_open.setObjectName("b_learn_open")
        self.hl_top.addWidget(self.b_learn_open, 0, QtCore.Qt.AlignTop)
        self.b_predict_open = QtWidgets.QPushButton(self.tab_osn)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_predict_open.sizePolicy().hasHeightForWidth())
        self.b_predict_open.setSizePolicy(sizePolicy)
        self.b_predict_open.setMinimumSize(QtCore.QSize(0, 40))
        self.b_predict_open.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_predict_open.setFont(font)
        self.b_predict_open.setStyleSheet("QPushButton{\n"
                                          "border-radius: 5px;\n"
                                          "border: 1px solid #2B2A29;\n"
                                          "padding: 10px 10px 10px 10px;\n"
                                          "color: black;\n"
                                          "}\n"
                                          "\n"
                                          "QPushButton:hover{\n"
                                          "background: solid #625B71;\n"
                                          "color: white;\n"
                                          "}")
        self.b_predict_open.setObjectName("b_predict_open")
        self.hl_top.addWidget(self.b_predict_open, 0, QtCore.Qt.AlignTop)
        self.b_info = QtWidgets.QToolButton(self.tab_osn)
        self.b_info.setMinimumSize(QtCore.QSize(40, 40))
        self.b_info.setMaximumSize(QtCore.QSize(40, 40))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_info.setFont(font)
        self.b_info.setToolTipDuration(-1)
        self.b_info.setStyleSheet("border-radius: 5px;\n"
                                  "border: 1px solid #2B2A29;\n"
                                  "color: black;\n"
                                  "")
        self.b_info.setObjectName("b_info")
        self.hl_top.addWidget(self.b_info, 0, QtCore.Qt.AlignVCenter)
        self.l_info_tr = QtWidgets.QLabel(self.tab_osn)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.l_info_tr.sizePolicy().hasHeightForWidth())
        self.l_info_tr.setSizePolicy(sizePolicy)
        self.l_info_tr.setMinimumSize(QtCore.QSize(73, 40))
        self.l_info_tr.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_info_tr.setFont(font)
        self.l_info_tr.setAcceptDrops(False)
        self.l_info_tr.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.l_info_tr.setStyleSheet("border-radius: 5px;\n"
                                     "border: 1px solid #2B2A29;\n"
                                     "padding: 10px 10px 10px 10px;\n"
                                     "color: black;\n"
                                     "")
        self.l_info_tr.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.l_info_tr.setObjectName("l_info_tr")
        self.hl_top.addWidget(self.l_info_tr, 0, QtCore.Qt.AlignTop)
        self.verticalLayout.addLayout(self.hl_top)
        self.top_line = QtWidgets.QFrame(self.tab_osn)
        self.top_line.setStyleSheet("background: #2B2A29;")
        self.top_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.top_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.top_line.setObjectName("top_line")
        self.verticalLayout.addWidget(self.top_line)
        self.hl_text_top = QtWidgets.QHBoxLayout()
        self.hl_text_top.setObjectName("hl_text_top")
        self.vl_name = QtWidgets.QVBoxLayout()
        self.vl_name.setObjectName("vl_name")
        self.l_name = QtWidgets.QLabel(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_name.setFont(font)
        self.l_name.setObjectName("l_name")
        self.vl_name.addWidget(self.l_name)
        self.cb_name = QtWidgets.QComboBox(self.tab_osn)
        self.cb_name.setMinimumSize(QtCore.QSize(0, 30))
        self.cb_name.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.cb_name.setFont(font)
        self.cb_name.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                   "border-radius: 5px;\n"
                                   "background: #D1D2FD solid;\n"
                                   "color: #2B2A29;\n"
                                   "")
        self.cb_name.setEditable(True)
        self.cb_name.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.cb_name.setIconSize(QtCore.QSize(20, 20))
        self.cb_name.setFrame(True)
        self.cb_name.setObjectName("cb_name")
        self.vl_name.addWidget(self.cb_name)
        self.hl_text_top.addLayout(self.vl_name)
        self.vl_x = QtWidgets.QVBoxLayout()
        self.vl_x.setObjectName("vl_x")
        self.l_gsx = QtWidgets.QLabel(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_gsx.setFont(font)
        self.l_gsx.setObjectName("l_gsx")
        self.vl_x.addWidget(self.l_gsx)
        self.le_gsx = QtWidgets.QLineEdit(self.tab_osn)
        self.le_gsx.setMinimumSize(QtCore.QSize(0, 30))
        self.le_gsx.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.le_gsx.setFont(font)
        self.le_gsx.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                  "border-radius: 5px;\n"
                                  "background: #D1D2FD solid;\n"
                                  "color: #2B2A29;\n"
                                  "")
        self.le_gsx.setInputMethodHints(QtCore.Qt.ImhHiddenText)
        self.le_gsx.setObjectName("le_gsx")
        self.vl_x.addWidget(self.le_gsx)
        self.hl_text_top.addLayout(self.vl_x)
        self.vl_y = QtWidgets.QVBoxLayout()
        self.vl_y.setObjectName("vl_y")
        self.l_gsy = QtWidgets.QLabel(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_gsy.setFont(font)
        self.l_gsy.setObjectName("l_gsy")
        self.vl_y.addWidget(self.l_gsy)
        self.le_gsy = QtWidgets.QLineEdit(self.tab_osn)
        self.le_gsy.setMinimumSize(QtCore.QSize(0, 30))
        self.le_gsy.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.le_gsy.setFont(font)
        self.le_gsy.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                  "border-radius: 5px;\n"
                                  "background: #D1D2FD solid;\n"
                                  "color: #2B2A29;\n"
                                  "")
        self.le_gsy.setInputMethodHints(QtCore.Qt.ImhNone)
        self.le_gsy.setObjectName("le_gsy")
        self.vl_y.addWidget(self.le_gsy)
        self.hl_text_top.addLayout(self.vl_y)
        self.vl_z = QtWidgets.QVBoxLayout()
        self.vl_z.setObjectName("vl_z")
        self.l_gsz = QtWidgets.QLabel(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_gsz.setFont(font)
        self.l_gsz.setObjectName("l_gsz")
        self.vl_z.addWidget(self.l_gsz)
        self.le_gsz = QtWidgets.QLineEdit(self.tab_osn)
        self.le_gsz.setMinimumSize(QtCore.QSize(0, 30))
        self.le_gsz.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.le_gsz.setFont(font)
        self.le_gsz.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                  "border-radius: 5px;\n"
                                  "background: #D1D2FD solid;\n"
                                  "color: #2B2A29;\n"
                                  "")
        self.le_gsz.setObjectName("le_gsz")
        self.vl_z.addWidget(self.le_gsz)
        self.hl_text_top.addLayout(self.vl_z)
        self.vl_cg = QtWidgets.QVBoxLayout()
        self.vl_cg.setObjectName("vl_cg")
        self.l_cg = QtWidgets.QLabel(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_cg.setFont(font)
        self.l_cg.setObjectName("l_cg")
        self.vl_cg.addWidget(self.l_cg)
        self.le_cg = QtWidgets.QLineEdit(self.tab_osn)
        self.le_cg.setMinimumSize(QtCore.QSize(0, 30))
        self.le_cg.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.le_cg.setFont(font)
        self.le_cg.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                 "border-radius: 5px;\n"
                                 "background: #D1D2FD solid;\n"
                                 "color: #2B2A29;\n"
                                 "")
        self.le_cg.setInputMethodHints(QtCore.Qt.ImhNone)
        self.le_cg.setObjectName("le_cg")
        self.vl_cg.addWidget(self.le_cg)
        self.hl_text_top.addLayout(self.vl_cg)
        self.hl_text_top.setStretch(0, 3)
        self.hl_text_top.setStretch(1, 1)
        self.hl_text_top.setStretch(2, 1)
        self.hl_text_top.setStretch(3, 1)
        self.hl_text_top.setStretch(4, 1)
        self.verticalLayout.addLayout(self.hl_text_top)
        self.l_mark = QtWidgets.QLabel(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_mark.setFont(font)
        self.l_mark.setObjectName("l_mark")
        self.verticalLayout.addWidget(self.l_mark)
        self.cb_mark = QtWidgets.QComboBox(self.tab_osn)
        self.cb_mark.setMinimumSize(QtCore.QSize(0, 30))
        self.cb_mark.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.cb_mark.setFont(font)
        self.cb_mark.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                   "border-radius: 5px;\n"
                                   "background: #D1D2FD solid;\n"
                                   "color: #2B2A29;\n"
                                   "")
        self.cb_mark.setEditable(True)
        self.cb_mark.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.cb_mark.setIconSize(QtCore.QSize(20, 20))
        self.cb_mark.setFrame(True)
        self.cb_mark.setObjectName("cb_mark")
        self.verticalLayout.addWidget(self.cb_mark)
        self.l_spf = QtWidgets.QLabel(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_spf.setFont(font)
        self.l_spf.setObjectName("l_spf")
        self.verticalLayout.addWidget(self.l_spf)
        self.cb_spf = QtWidgets.QComboBox(self.tab_osn)
        self.cb_spf.setMinimumSize(QtCore.QSize(0, 30))
        self.cb_spf.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.cb_spf.setFont(font)
        self.cb_spf.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                  "border-radius: 5px;\n"
                                  "background: #D1D2FD solid;\n"
                                  "color: #2B2A29;\n"
                                  "")
        self.cb_spf.setEditable(True)
        self.cb_spf.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.cb_spf.setIconSize(QtCore.QSize(20, 20))
        self.cb_spf.setFrame(True)
        self.cb_spf.setObjectName("cb_spf")
        self.verticalLayout.addWidget(self.cb_spf)
        self.l_tt = QtWidgets.QLabel(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_tt.setFont(font)
        self.l_tt.setObjectName("l_tt")
        self.verticalLayout.addWidget(self.l_tt)
        self.cb_tt = QtWidgets.QComboBox(self.tab_osn)
        self.cb_tt.setMinimumSize(QtCore.QSize(0, 30))
        self.cb_tt.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.cb_tt.setFont(font)
        self.cb_tt.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.cb_tt.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                 "border-radius: 5px;\n"
                                 "background: #D1D2FD solid;\n"
                                 "color: #2B2A29;\n"
                                 "")
        self.cb_tt.setEditable(True)
        self.cb_tt.setCurrentText("")
        self.cb_tt.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.cb_tt.setIconSize(QtCore.QSize(20, 20))
        self.cb_tt.setFrame(True)
        self.cb_tt.setObjectName("cb_tt")
        self.verticalLayout.addWidget(self.cb_tt)
        self.b_ready = QtWidgets.QPushButton(self.tab_osn)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_ready.sizePolicy().hasHeightForWidth())
        self.b_ready.setSizePolicy(sizePolicy)
        self.b_ready.setMaximumSize(QtCore.QSize(16777215, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_ready.setFont(font)
        self.b_ready.setStyleSheet("QPushButton{\n"
                                   "border-radius: 5px;\n"
                                   "border: 1px solid #2B2A29;\n"
                                   "color: black;\n"
                                   "padding: 10px 10px 10px 10px;\n"
                                   "}\n"
                                   "\n"
                                   "QPushButton:hover{\n"
                                   "background: solid #625B71;\n"
                                   "color: white;\n"
                                   "}")
        self.b_ready.setObjectName("b_ready")
        self.verticalLayout.addWidget(self.b_ready, 0, QtCore.Qt.AlignRight)
        self.hl_out_header = QtWidgets.QHBoxLayout()
        self.hl_out_header.setObjectName("hl_out_header")
        self.l_out = QtWidgets.QLabel(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_out.setFont(font)
        self.l_out.setObjectName("l_out")
        self.hl_out_header.addWidget(self.l_out)
        self.b_clear_in = QtWidgets.QPushButton(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_clear_in.setFont(font)
        self.b_clear_in.setStyleSheet("QPushButton{\n"
                                      "border-radius: 5px;\n"
                                      "border: 1px solid #2B2A29;\n"
                                      "color: black;\n"
                                      "padding: 3px 10px 3px 10px;\n"
                                      "}\n"
                                      "\n"
                                      "QPushButton:hover{\n"
                                      "background: solid #625B71;\n"
                                      "color: white;\n"
                                      "}")
        self.b_clear_in.setObjectName("b_clear_in")
        self.hl_out_header.addWidget(self.b_clear_in, 0, QtCore.Qt.AlignRight)
        self.b_clear_out = QtWidgets.QPushButton(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_clear_out.setFont(font)
        self.b_clear_out.setStyleSheet("QPushButton{\n"
                                       "border-radius: 5px;\n"
                                       "border: 1px solid #2B2A29;\n"
                                       "color: black;\n"
                                       "padding: 3px 10px 3px 10px;\n"
                                       "}\n"
                                       "\n"
                                       "QPushButton:hover{\n"
                                       "background: solid #625B71;\n"
                                       "color: white;\n"
                                       "}")
        self.b_clear_out.setObjectName("b_clear_out")
        self.hl_out_header.addWidget(self.b_clear_out, 0, QtCore.Qt.AlignRight)
        self.hl_out_header.setStretch(0, 5)
        self.verticalLayout.addLayout(self.hl_out_header)
        self.bottom_line = QtWidgets.QFrame(self.tab_osn)
        self.bottom_line.setStyleSheet("background: #2B2A29;")
        self.bottom_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.bottom_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.bottom_line.setObjectName("bottom_line")
        self.verticalLayout.addWidget(self.bottom_line)
        self.tb_output = QtWidgets.QPlainTextEdit(self.tab_osn)
        self.tb_output.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tb_output.sizePolicy().hasHeightForWidth())
        self.tb_output.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.tb_output.setFont(font)
        self.tb_output.setStyleSheet("border-radius: 5px;\n"
                                     "border: 1px solid  rgb(155, 241, 56);\n"
                                     "color: black;\n"
                                     "")
        self.tb_output.setReadOnly(True)
        self.tb_output.setPlainText("")
        self.tb_output.setObjectName("tb_output")
        self.verticalLayout.addWidget(self.tb_output)
        self.pb_fit_progress = QtWidgets.QProgressBar(self.tab_osn)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.pb_fit_progress.setFont(font)
        self.pb_fit_progress.setStyleSheet("border-radius: 5px;\n"
                                           "border: 1px solid  rgb(155, 241, 56);\n"
                                           "color: black;\n"
                                           "")
        self.pb_fit_progress.setProperty("value", 0)
        self.pb_fit_progress.setAlignment(QtCore.Qt.AlignCenter)
        self.pb_fit_progress.setObjectName("pb_fit_progress")
        self.verticalLayout.addWidget(self.pb_fit_progress)
        self.tabs.addTab(self.tab_osn, "")
        self.tab_usl = QtWidgets.QWidget()
        self.tab_usl.setObjectName("tab_usl")
        self.tabs.addTab(self.tab_usl, "")
        self.verticalLayout_2.addWidget(self.tabs)
        main_window.setCentralWidget(self.central)
        self.usl = QtWidgets.QAction(main_window)
        self.usl.setObjectName("usl")

        self.retranslate_ui(main_window)
        self.tabs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslate_ui(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "MainWindow"))
        self.b_learn_open.setText(_translate("main_window", "Выбрать данные для обучения"))
        self.b_predict_open.setText(_translate("main_window", "Выбрать данные для прогнозирования"))
        self.b_info.setToolTip(_translate("main_window", "Информация"))
        self.b_info.setText(_translate("main_window", "..."))
        self.l_info_tr.setToolTip(_translate("main_window", "Время обучения модели"))
        self.l_info_tr.setText(_translate("main_window", "0"))
        self.l_name.setText(_translate("main_window", "name"))
        self.cb_name.setToolTip(_translate("main_window", "Технические требования"))
        self.l_gsx.setText(_translate("main_window", "gs_x"))
        self.le_gsx.setToolTip(_translate("main_window", "Размер X"))
        self.l_gsy.setText(_translate("main_window", "gs_y"))
        self.le_gsy.setToolTip(_translate("main_window", "Размер Y"))
        self.l_gsz.setText(_translate("main_window", "gs_z"))
        self.le_gsz.setToolTip(_translate("main_window", "Размер Z"))
        self.l_cg.setText(_translate("main_window", "cg"))
        self.le_cg.setToolTip(_translate("main_window", "Конструктивная группа (две цифры)"))
        self.l_mark.setText(_translate("main_window", "mark"))
        self.cb_mark.setToolTip(_translate("main_window", "Технические требования"))
        self.l_spf.setText(_translate("main_window", "spf"))
        self.cb_spf.setToolTip(_translate("main_window", "Технические требования"))
        self.l_tt.setText(_translate("main_window", "tt"))
        self.cb_tt.setToolTip(_translate("main_window", "Технические требования"))
        self.b_ready.setText(_translate("main_window", "Готово"))
        self.l_out.setText(_translate("main_window", "output"))
        self.b_clear_in.setText(_translate("main_window", "clear input"))
        self.b_clear_out.setText(_translate("main_window", "clear output"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_osn), _translate("main_window", "osn"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_usl), _translate("main_window", "usl"))
        self.usl.setText(_translate("main_window", "usl"))
