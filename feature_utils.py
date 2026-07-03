"""Функція побудови фіч для моделі прогнозу швидкості.
Використовувати ІДЕНТИЧНО до того, як рахувалось при тренуванні —
інакше model.predict() дасть сміттєвий результат.
"""
import numpy as np

FEATURE_NAMES = [
    'v1', 'v2', 'v3', 'v4', 'v5',
    'd1', 'd2', 'd3', 'd4',
    'mean_v', 'std_v',
    'trend_slope',
]


def build_features(window):
    """window: список з 5 останніх швидкостей (км/год), від найстарішого до найновішого.
    Повертає список з 12 чисел у порядку FEATURE_NAMES.
    """
    window = list(window)
    diffs = [window[i + 1] - window[i] for i in range(len(window) - 1)]
    mean_v = float(np.mean(window))
    std_v = float(np.std(window))
    trend_slope = float(np.polyfit(range(len(window)), window, 1)[0])
    return window + diffs + [mean_v, std_v, trend_slope]
