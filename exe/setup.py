from cx_Freeze import setup, Executable

setup(
    name="TechRotePredictor",
    version="1.0",
    description="Программа для прогнозирования технологического маршрута",
    executables=[Executable("core/main.py", base="Win32GUI")]
)
