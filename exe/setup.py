from cx_Freeze import setup, Executable

options = {
    'build_exe': {
        'packages': ['sklearn'],
    },
}

setup(
    name="TechRotePredictor",
    version="1.0",
    description="Программа для прогнозирования технологического маршрута",
    options=options,
    executables=[
        Executable(
            "D:/College/Other/PyProjects/PyTechnologicalRoute/exe/TechRoutePredictor.py",
            base="Win32GUI"
        )
    ]
)
