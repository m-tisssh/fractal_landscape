import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import zoom
import json
from tkinter import ttk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

# Файл для сохранения настроек палитры
PALETTE_FILE = "palette_settings.json"

# # Цвета для биомов по умолчанию
# DEFAULT_PALETTE = {
#     "Ocean Start": "#000080",
#     "Ocean End": "#0000FF",
#     "Sea Start": "#00BFFF",
#     "Sea End": "#87CEEB",
#     "Plains Start": "#228B22",
#     "Plains End": "#ADFF2F",
#     "Mountains Start": "#8B4513",
#     "Mountains End": "#DEB887",
#     "Snowy Mountains Start": "#c2ad91",
#     "Snowy Mountains End": "#F8F8FF",
# }
DEFAULT_PALETTE = {
    "Ocean Start": "#1E3A56",   # Темный, глубокий оттенок синего
    "Ocean End": "#4A90D9",      # Умеренно яркий синий цвет
    "Sea Start": "#7EB5B5",      # Тихий, мутный оттенок морской воды
    "Sea End": "#A6D8D8",        # Светлый и нежный оттенок моря
    "Plains Start": "#A4C639",   # Спокойный зеленый, напоминающий траву
    "Plains End": "#B0D94A",     # Легкий оттенок весной зелени
    "Mountains Start": "#6F4F28", # Сдержанный коричневый, напоминающий землю
    "Mountains End": "#C8B29A",   # Мягкий песочный оттенок
    "Snowy Mountains Start": "#C4B89F", # Легкий оттенок снега с пыльным оттенком
    "Snowy Mountains End": "#F1F8FC",   # Почти белый, нежный с голубоватым оттенком
}
# DEFAULT_PALETTE = {
#     "Ocean Start": "#003366",   # Глубокий и насыщенный темно-синий
#     "Ocean End": "#00BFFF",     # Яркий бирюзовый, напоминающий экзотическое море
#     "Sea Start": "#FF4500",     # Яркий оранжево-красный, напоминающий закат
#     "Sea End": "#FFD700",       # Ярко-желтый, как солнечные лучи
#     "Plains Start": "#FF6347",  # Насыщенный томатный красный
#     "Plains End": "#FF8C00",    # Яркий оранжевый
#     "Mountains Start": "#800080", # Интенсивный пурпурный
#     "Mountains End": "#9400D3",   # Яркий фиолетовый
#     "Snowy Mountains Start": "#FF1493",  # Яркий розовый, напоминающий утренний свет
#     "Snowy Mountains End": "#DDA0DD",    # Лаванда, с фиолетовым оттенком
# }




# Diamond-Square algorithm для создания рельефа местности
def diamond_square(size, roughness, corners):
    size = 2 ** size + 1
    terrain = np.zeros((size, size))

    # Set the corners
    terrain[0, 0], terrain[0, -1], terrain[-1, 0], terrain[-1, -1] = corners

    step = size - 1
    while step > 1:
        half_step = step // 2

        # Шаг "Diamond"
        for y in range(0, size - 1, step):
            for x in range(0, size - 1, step):
                avg = (terrain[y, x] + terrain[y, x + step] +
                       terrain[y + step, x] + terrain[y + step, x + step]) / 4
                terrain[y + half_step, x + half_step] = avg + np.random.uniform(-roughness, roughness)

        # Шаг "Square"
        for y in range(0, size, half_step):
            for x in range((y + half_step) % step, size, step):
                neighbors = []
                if y - half_step >= 0:
                    neighbors.append(terrain[y - half_step, x])
                if y + half_step < size:
                    neighbors.append(terrain[y + half_step, x])
                if x - half_step >= 0:
                    neighbors.append(terrain[y, x - half_step])
                if x + half_step < size:
                    neighbors.append(terrain[y, x + half_step])
                avg = sum(neighbors) / len(neighbors)
                terrain[y, x] = avg + np.random.uniform(-roughness, roughness)

        step //= 2
        roughness /= 2

    # Нормализация значений и перевод в процентное соотношение
    terrain -= terrain.min()
    terrain /= terrain.max()
    terrain *= 100
    
    return terrain


# triangle_division algorithm для создания рельефа местности
def triangle_division(size, roughness, corners):
    """
    Алгоритм разбиения треугольников для создания рельефа.

    :param size: количество итераций для разбиения треугольников.
    :param roughness: Параметр шероховатости, управляющий изменением высоты.
    :param corners: Начальные высоты углов треугольника.
    :return: двумерный числовой массив, представляющий местность.
    """
    # Инициализируем сетку для сохранения высот
    grid_size = 2**size + 1
    terrain = np.full((grid_size, grid_size), np.nan)

    # Установить начальные углы
    terrain[0, 0] = corners[0]
    terrain[0, -1] = corners[1]
    terrain[-1, 0] = corners[2]
    terrain[-1, -1] = corners[3]

    # Определение треугольников для обработки
    triangles = [
        [(0, 0), (0, grid_size - 1), (grid_size - 1, 0)],  # Left triangle
        [(0, grid_size - 1), (grid_size - 1, grid_size - 1), (grid_size - 1, 0)],  # Right triangle
    ]

    # Повторять каждый уровень детализации
    for step in range(size):
        new_triangles = []
        k = (grid_size - 1) / (2**step)  # Scale factor
        displacement = k / 2  # Maximum displacement based on current triangle size

        for tri in triangles:
            p1, p2, p3 = tri

            # Вычисление средних точек
            mid1 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            mid2 = ((p2[0] + p3[0]) // 2, (p2[1] + p3[1]) // 2)
            mid3 = ((p3[0] + p1[0]) // 2, (p3[1] + p1[1]) // 2)

            # Вычисление высот для средних точек
            for mid, pts in [(mid1, (p1, p2)), (mid2, (p2, p3)), (mid3, (p3, p1))]:
                if np.isnan(terrain[mid]):
                    avg_height = np.mean([terrain[pts[0]], terrain[pts[1]]])
                    perturbation = np.random.uniform(-displacement, displacement) * roughness
                    terrain[mid] = avg_height + perturbation

            # Разделение треугольника на четыре меньших треугольника
            new_triangles.extend([
                [p1, mid1, mid3],
                [mid1, p2, mid2],
                [mid3, mid2, p3],
                [mid1, mid2, mid3],
            ])

        # Обновление треугольников для следующей итерации 
        triangles = new_triangles

    # Заполнить все оставшиеся значения NaN (например, на границах).
    terrain[np.isnan(terrain)] = 0

    return terrain


class AdvancedSettingsWindow(tk.Toplevel):
    def __init__(self, master, extreme_points, callback):
        super().__init__(master)
        self.title("Advanced Settings")
        self.geometry("500x400")
        self.extreme_points = extreme_points
        self.callback = callback  # Функция для передачи изменений в главное приложение

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Вкладка 1: Диапазоны биомов
        self.biome_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.biome_tab, text="Biome Ranges")

        # Вкладка 2: Настройки визуализации
        self.visual_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visual_tab, text="Visualization")

        # Вкладка 3: Другие настройки
        self.other_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.other_tab, text="Other")

        self.create_biome_settings()
        self.create_visual_settings()
        self.create_other_settings()

        # Кнопки для Save, Reset, and Close
        ttk.Button(self, text="Save", command=self.save_settings).pack(side="left", padx=10, pady=10)
        ttk.Button(self, text="Reset", command=self.reset_settings).pack(side="left", padx=10, pady=10)
        ttk.Button(self, text="Close", command=self.destroy).pack(side="right", padx=10, pady=10)

    def create_biome_settings(self):
        """Создавайте виджеты для настройки диапазона биомов."""
        ttk.Label(self.biome_tab, text="Biome Elevation Ranges", font=("Arial", 12, "bold")).pack(pady=10)
        frame = ttk.Frame(self.biome_tab)
        frame.pack(fill="x", padx=10, pady=10)

        self.range_vars = {}
        for i, (biome, (min_val, max_val)) in enumerate(self.extreme_points.items()):
            ttk.Label(frame, text=biome).grid(row=i, column=0, sticky="w")
            min_var = tk.DoubleVar(value=min_val)
            max_var = tk.DoubleVar(value=max_val)
            ttk.Entry(frame, textvariable=min_var, width=10).grid(row=i, column=1, padx=5)
            ttk.Entry(frame, textvariable=max_var, width=10).grid(row=i, column=2, padx=5)
            self.range_vars[biome] = (min_var, max_var)

    def create_visual_settings(self):
        """Создать виджеты для настройки визуализации."""
        ttk.Label(self.visual_tab, text="Visualization Settings", font=("Arial", 12, "bold")).pack(pady=10)
        frame = ttk.Frame(self.visual_tab)
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Map Brightness").grid(row=0, column=0, sticky="w")
        self.brightness_var = tk.DoubleVar(value=1.0)
        tk.Scale(frame, variable=self.brightness_var, from_=0.5, to=2.0, resolution=0.1, orient="horizontal").grid(row=0, column=1, sticky="ew", padx=10)

        ttk.Label(frame, text="Map Contrast").grid(row=1, column=0, sticky="w")
        self.contrast_var = tk.DoubleVar(value=1.0)
        tk.Scale(frame, variable=self.contrast_var, from_=0.5, to=2.0, resolution=0.1, orient="horizontal").grid(row=1, column=1, sticky="ew", padx=10)

    def create_other_settings(self):
        """Создать виджеты для других настроек."""
        ttk.Label(self.other_tab, text="Other Settings", font=("Arial", 12, "bold")).pack(pady=10)
        frame = ttk.Frame(self.other_tab)
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Smoothness (Blur Radius)").grid(row=0, column=0, sticky="w")
        self.smoothness_var = tk.IntVar(value=0)
        tk.Scale(frame, variable=self.smoothness_var, from_=0, to=10, resolution=1, orient="horizontal").grid(row=0, column=1, sticky="ew", padx=10)

        ttk.Label(frame, text="Enable Grid").grid(row=1, column=0, sticky="w")
        self.grid_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, variable=self.grid_enabled_var).grid(row=1, column=1, sticky="w")

    def save_settings(self):
        """Сохранить настройки и отправьте их в основное приложение."""
        # Собираем все настройки
        for biome, (min_var, max_var) in self.range_vars.items():
            self.extreme_points[biome] = (min_var.get(), max_var.get())

        settings = {
            "extreme_points": self.extreme_points,
            "brightness": self.brightness_var.get(),
            "contrast": self.contrast_var.get(),
            "smoothness": self.smoothness_var.get(),
            "grid_enabled": self.grid_enabled_var.get(),
        }

        # Передаем настройки через callback в главное приложение
        self.callback(settings)  # Передаем настройки обратно
        messagebox.showinfo("Success", "Settings saved successfully!")


    def reset_settings(self):
        """Сброс всех настроек до значений по умолчанию."""
        self.extreme_points = {
            "Ocean": (-3000, -1001),
            "Sea Shore": (-1000, -1),
            "Plains": (0, 999),
            "Mountains": (1000, 2899),
            "Snowy Mountains": (2900, 3000),
        }

        for biome, (min_var, max_var) in self.range_vars.items():
            min_var.set(self.extreme_points[biome][0])
            max_var.set(self.extreme_points[biome][1])

        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0) 
        self.smoothness_var.set(0)
        self.grid_enabled_var.set(False)


# Класс для настроек палитры
class PaletteSettings:
    def __init__(self, master, update_palette_callback):
        self.master = master
        self.update_palette_callback = update_palette_callback
        self.master.title("Biome Palette Settings")
        self.palette = self.load_palette()

        self.color_labels = []
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.master, text="Biome Palette Settings", font=("Arial", 14)).pack(pady=10)

        self.colors_frame = tk.Frame(self.master)
        self.colors_frame.pack(pady=10)

        # Создание widgets для каждого цвета
        for i, (label, color) in enumerate(self.palette.items()):
            row = i // 2
            col = i % 2
            frame = tk.Frame(self.colors_frame, bd=2, relief="ridge", padx=10, pady=10)
            frame.grid(row=row, column=col, padx=10, pady=10)

            tk.Label(frame, text=label, font=("Arial", 10)).pack()
            color_label = tk.Label(frame, bg=color, width=20, height=2)
            color_label.pack(pady=5)

            # Привязать событие к изменению цвета
            color_label.bind("<Button-1>", lambda e, lbl=label: self.change_color(lbl))

            self.color_labels.append((label, color_label))

        tk.Button(self.master, text="Save", command=self.save_palette, font=("Arial", 12)).pack(pady=10)
        
        # Кнопка сброса
        tk.Button(self.master, text="Reset", command=self.reset_palette, font=("Arial", 12)).pack(pady=10)

    def change_color(self, label):
        # Открытие диалогового окна выбора цвета
        color_code = colorchooser.askcolor(title=f"Choose color for {label}")[1]
        if color_code:
            self.palette[label] = color_code
            for lbl, widget in self.color_labels:
                if lbl == label:
                    widget.config(bg=color_code)

    def save_palette(self):
        try:
            with open(PALETTE_FILE, "w") as file:
                json.dump(self.palette, file)
            self.update_palette_callback(self.palette)  # Передача новых настроек в основное приложение
            messagebox.showinfo("Success", "Palette settings saved!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def reset_palette(self):
        self.palette = DEFAULT_PALETTE.copy()
        for lbl, widget in self.color_labels:
            widget.config(bg=self.palette[lbl])

    def load_palette(self):
        try:
            with open(PALETTE_FILE, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return DEFAULT_PALETTE.copy()
        except json.JSONDecodeError:
            messagebox.showwarning("Error", "Failed to load palette. Using default settings.")
            return DEFAULT_PALETTE.copy()


def adjust_brightness(image_array, factor):
    """Adjust brightness of an image."""
    return np.clip(image_array * factor, 0, 255).astype(np.uint8)

def adjust_contrast(image_array, factor):
    """Adjust contrast of an image."""
    mean = np.mean(image_array, axis=(0, 1), keepdims=True)
    return np.clip((image_array - mean) * factor + mean, 0, 255).astype(np.uint8)

def gaussian_blur(image_array, radius):
        """Реализация размытия по Гауссу с использованием NumPy."""
        def gaussian_kernel(size, sigma):
            """Создаем ядро Гаусса."""
            ax = np.arange(-size // 2 + 1, size // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            return kernel / np.sum(kernel)
        
        # Рассчитываем размер ядра и создаем его
        kernel_size = int(2 * radius + 1)
        kernel = gaussian_kernel(kernel_size, radius)
        
        # Добавляем паддинг к изображению для обработки краев
        pad_width = kernel_size // 2
        padded_image = np.pad(image_array, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')
        
        # Применяем ядро к каждому каналу
        blurred_image = np.zeros_like(image_array)
        for i in range(3):  # Для R, G, B
            for y in range(image_array.shape[0]):
                for x in range(image_array.shape[1]):
                    region = padded_image[y:y + kernel_size, x:x + kernel_size, i]
                    blurred_image[y, x, i] = np.sum(region * kernel)
        
        return blurred_image.astype(np.uint8)
    

# Класс для генерации ландшафта
class FractalLandscapeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fractal Landscape Generator")
        self.geometry("900x700")
        self.palette = DEFAULT_PALETTE.copy()

        # Attributes to store the latest generated terrain and biomes
        self.last_terrain = None
        self.last_biomes = None

        # Initial settings for terrain generation
        self.size_var = tk.IntVar(value=7)
        self.roughness_var = tk.DoubleVar(value=1.0)
        self.resolution_var = tk.IntVar(value=256)
        self.corner_vars = [tk.DoubleVar(value=0) for _ in range(4)]

        self.brightness = 1.0  # Значение по умолчанию
        self.contrast = 1.0    # Значение по умолчанию
        self.smoothness = 0.0  # Значение по умолчанию
        self.extreme_points = {}  # Биомы и их границы
        self.grid_enabled = False

        self.default_extreme_points = {
            "Ocean": (-3000, -1001),
            "Sea Shore": (-1000, -1),
            "Plains": (0, 999),
            "Mountains": (1000, 2899),
            "Snowy Mountains": (2900, 3000)
        }
        self.extreme_points = self.default_extreme_points.copy()

        # Дополнительные атрибуты для масштабирования
        self.scale_factor = 2000.0

        self.create_widgets()

    def create_widgets(self):
        # Рамка для элементов управления
        control_frame = ttk.Frame(self)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        # Создание виджетов для определения параметров рельефа
        for i, label in enumerate(["Top-Left (m)", "Top-Right (m)", "Bottom-Left (m)", "Bottom-Right (m)"]):
            ttk.Label(control_frame, text=f"{label}").grid(row=i, column=0, sticky="w")
            ttk.Spinbox(control_frame, from_=-100, to=100, increment=1, textvariable=self.corner_vars[i], width=10).grid(row=i, column=1)

        ttk.Label(control_frame, text="Roughness").grid(row=4, column=0, sticky="w")
        tk.Scale(control_frame, variable=self.roughness_var, from_=0.1, to=2.0, resolution=0.1, orient="horizontal").grid(row=4, column=1, sticky="ew")

        ttk.Label(control_frame, text="Grid Size").grid(row=5, column=0, sticky="w")
        tk.Spinbox(control_frame, from_=4, to=10, textvariable=self.size_var).grid(row=5, column=1)

        ttk.Label(control_frame, text="Resolution (pixels)").grid(row=6, column=0, sticky="w")
        tk.Scale(control_frame, variable=self.resolution_var, from_=128, to=1024, resolution=64, orient="horizontal").grid(row=6, column=1, sticky="ew")

        ttk.Button(control_frame, text="Start", command=self.generate).grid(row=7, column=0, sticky="ew", columnspan=2)
        ttk.Button(control_frame, text="Clear", command=self.clear).grid(row=8, column=0, sticky="ew", columnspan=2)
        ttk.Button(control_frame, text="Save", command=self.save_image).grid(row=9, column=0, sticky="ew", columnspan=2)
        ttk.Button(control_frame, text="Advanced Settings", command=self.open_additional_settings).grid(row=10, column=0, sticky="ew", columnspan=2)
        ttk.Button(control_frame, text="Palette Settings", command=self.open_palette_settings).grid(row=11, column=0, pady=5)

        self.algorithm_var = tk.StringVar(value="Diamond-Square")
        ttk.Label(control_frame, text="Algorithm").grid(row=12, column=0, sticky="w")
        ttk.OptionMenu(control_frame, self.algorithm_var, "Diamond-Square", "Diamond-Square", "Triangle Division").grid(row=12, column=1)

        ttk.Button(control_frame, text="Show 3D Plot", command=self.show_3d_plot).grid(row=13, column=0, sticky="ew", columnspan=2)

        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Привязка событий для увеличения и уменьшения
        self.canvas_widget.bind("<Enter>", self.enable_zoom_controls)
        self.canvas_widget.bind("<Leave>", self.disable_zoom_controls)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def show_3d_plot(self):
        """Откройте новое окно для отображения 3D-графика поверхности местности."""
        if self.last_terrain is None:
            messagebox.showerror("Error", "Generate terrain first!")
            return

        # Создание окна для 3D графика
        plot_window = tk.Toplevel(self)
        plot_window.title("3D Surface Plot")
        plot_window.geometry("800x600")

        # Подготовка данных
        terrain = self.last_terrain
        scale_factor = 2  # Масштабирование для гладкости
        z = zoom(terrain, scale_factor, order=1)  # Интерполяция высот

        # Пересчитываем сетку x и y на основе нового размера z
        x = np.linspace(0, z.shape[1] - 1, z.shape[1])
        y = np.linspace(0, z.shape[0] - 1, z.shape[0])
        x, y = np.meshgrid(x, y)

        # Используем карту цветов
        cmap = LinearSegmentedColormap.from_list(
            "custom_palette",
            list(self.palette.values())  # Преобразуем палитру в список цветов
        )

        # Создаем 3D график
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Построение поверхности
        surface = ax.plot_surface(x, y, z, cmap=cmap, edgecolor='none', antialiased=True)

        # Отзеркаливание оси Y
        ax.invert_yaxis()

        # Добавляем цветовую шкалу
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, pad=0.1)

        # Добавляем подписи
        ax.set_title("3D Surface Plot with Custom Palette", pad=20)
        ax.set_xlabel("X", labelpad=10)
        ax.set_ylabel("Y", labelpad=10)
        ax.set_zlabel("meters", labelpad=10)

        # Встраиваем график в окно Tkinter
        canvas = FigureCanvasTkAgg(fig, plot_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        canvas.draw()


    def generate(self):
        size = self.size_var.get()
        roughness = self.roughness_var.get()
        resolution = self.resolution_var.get()
        corners = [var.get() for var in self.corner_vars]

        # Select algorithm
        if self.algorithm_var.get() == "Diamond-Square":
            terrain = diamond_square(size, roughness, corners)
        elif self.algorithm_var.get() == "Triangle Division":
            terrain = triangle_division(size, roughness, corners)
        else:
            raise ValueError("Unknown algorithm selected")


        # Нормализация значений
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 6000 - 3000

        scale_factor = resolution / terrain.shape[0]
        terrain = zoom(terrain, scale_factor, order=1)

        # Classify the terrain into biomes
        biomes = self.classify_biomes(terrain)

        # Save terrain and biomes for re-plotting
        self.last_terrain = terrain
        self.last_biomes = biomes

        # Update display
        # self.ax.clear()
        self.plot_terrain(terrain, biomes)
        self.canvas.draw()
        
    def update_palette(self, new_palette):
        """При необходимости обновите палитру и дисплей."""
        self.palette = new_palette
        
        # Если карта уже была сгенерирована, перерисовать её с новой палитрой
        if self.last_terrain is not None and self.last_biomes is not None:
            self.plot_terrain(self.last_terrain, self.last_biomes)
            self.canvas.draw()


    def classify_biomes(self, terrain):
        """Классифицируйте биомы по высоте над уровнем моря"""
        biomes = np.empty_like(terrain, dtype='<U16')

        biomes[(terrain >= -3000) & (terrain < -1000)] = "Ocean"
        biomes[(terrain >= -1000) & (terrain < 0)] = "Sea"
        biomes[(terrain >= 0) & (terrain < 1000)] = "Plains"
        biomes[(terrain >= 1000) & (terrain < 2800)] = "Mountains"
        biomes[terrain >= 2800] = "Snowy Mountains"

        return biomes

    def plot_terrain(self, terrain, biomes):
        """Нарисуйте рельеф местности с помощью цветов, основанных на палитре и расширенных настройках."""
        rgb_array = np.zeros((*terrain.shape, 3), dtype=np.uint8)

        # Цветовые диапазоны биомов
        biome_color_ranges = {
            "Ocean": (self.palette["Ocean Start"], self.palette["Ocean End"]),
            "Sea": (self.palette["Sea Start"], self.palette["Sea End"]),
            "Plains": (self.palette["Plains Start"], self.palette["Plains End"]),
            "Mountains": (self.palette["Mountains Start"], self.palette["Mountains End"]),
            "Snowy Mountains": (self.palette["Snowy Mountains Start"], self.palette["Snowy Mountains End"]),
        }

        for biome, (start_hex, end_hex) in biome_color_ranges.items():
            start_rgb = np.array([int(start_hex[i:i + 2], 16) for i in (1, 3, 5)])
            end_rgb = np.array([int(end_hex[i:i + 2], 16) for i in (1, 3, 5)])

            mask = biomes == biome
            if mask.any():
                biome_min = terrain[mask].min()
                biome_max = terrain[mask].max()
                normalized_values = (terrain[mask] - biome_min) / (biome_max - biome_min)
                rgb_values = (start_rgb + (end_rgb - start_rgb) * normalized_values[:, None]).astype(np.uint8)
                rgb_array[mask] = rgb_values

        # Применяем настройки яркости и контраста
        rgb_array = adjust_brightness(rgb_array, self.brightness)
        rgb_array = adjust_contrast(rgb_array, self.contrast)

        # Применяем размытие
        if self.smoothness > 0:
            rgb_array = gaussian_blur(rgb_array, self.smoothness)

        # Постройте изображение
        self.ax.clear()
        self.ax.imshow(rgb_array, interpolation="nearest")
        self.ax.axis("off")

        # Рисовать сетку, если она включена
        if self.grid_enabled:
            rows, cols = terrain.shape
            self.ax.set_xticks(np.arange(0, cols, 1), minor=False)
            self.ax.set_yticks(np.arange(0, rows, 1), minor=False)
            self.ax.grid(which="major", color="grey", linestyle="-", linewidth=10)
            self.ax.tick_params(which="major", size=0)


    def clear(self):
        self.ax.clear()
        self.canvas.draw()

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if not file_path:
            return

        size = self.resolution_var.get()
        self.figure.set_size_inches(size / 100, size / 100)
        self.figure.savefig(file_path, dpi=100)

    def apply_advanced_settings(self, settings):
        """Обратный вызов для получения изменений из окна дополнительных настроек."""
        self.extreme_points = settings["extreme_points"]
        self.brightness = settings["brightness"]
        self.contrast = settings["contrast"]
        self.smoothness = settings["smoothness"]
        self.grid_enabled = settings["grid_enabled"]

        # Перерисовка текущего ландшафта с новыми настройками
        if hasattr(self, "last_terrain"):
            self.plot_terrain(self.last_terrain, self.last_biomes)
            self.canvas.draw()


    def open_additional_settings(self):
        AdvancedSettingsWindow(self, self.extreme_points, self.apply_advanced_settings)

    def reset_to_default(self, settings_window):
        """Сбросить настройки на значения по умолчанию и закрыть окно"""
        self.extreme_points = self.default_extreme_points.copy()
        self.update_extreme_points_display()

        # Закрыть окно дополнительных настроек
        settings_window.destroy()

    def update_extreme_points_display(self):
        """Обновить отображение экстремальных точек"""
        for biome, (min_val, max_val) in self.extreme_points.items():
            print(f"{biome}: Min = {min_val}, Max = {max_val}")

    def open_palette_settings(self):
        palette_window = tk.Toplevel(self)
        PaletteSettings(palette_window, self.update_palette)

    def enable_zoom_controls(self, event):
        """Включение управления масштабированием при наведении на изображение."""
        self.bind("+", self.zoom_in)
        self.bind("-", self.zoom_out)

    def disable_zoom_controls(self, event):
        """Отключение управления масштабированием при выходе из области изображения."""
        self.unbind("+")
        self.unbind("-")

    def zoom_in(self, event=None):
        """Увеличение масштаба изображения."""
        self.scale_factor *= 1.1
        self.redraw_image()

    def zoom_out(self, event=None):
        """Уменьшение масштаба изображения."""
        self.scale_factor /= 1.1
        self.redraw_image()

    def redraw_image(self):
        """Перерисовка изображения с учетом текущего масштаба."""
        if self.last_terrain is None:
            return

        # Применяем масштабирование к terrain
        zoomed_terrain = zoom(self.last_terrain, self.scale_factor, order=1)

        # Пересчитываем biomes на основе масштабированного terrain
        zoomed_biomes = self.classify_biomes(zoomed_terrain)

        # Обновляем отображение
        self.plot_terrain(zoomed_terrain, zoomed_biomes)
        self.canvas.draw()


# Run the application
if __name__ == "__main__":
    app = FractalLandscapeApp()
    app.mainloop()
