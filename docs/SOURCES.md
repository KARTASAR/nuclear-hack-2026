# Источники И Научное Обоснование Решения

Этот файл фиксирует, на какие статьи/ресурсы мы опирались и что именно из них использовали в проекте.

## 1) Что реально взяли в итоговый подход

| Направление | Что взяли в работу | Статус |
|---|---|---|
| Preprocessing-пайплайн | Удаление спайков, baseline-correction, нормализация, harmonization по общей wave-сетке | Внедрено |
| Baseline correction | Семейство penalized least squares (`airPLS`, `arPLS`) | Внедрено |
| Нормализация | `SNV`, `MSC`, `EMSC` как сравниваемые варианты | Внедрено |
| Validation protocol | Group-wise split по мышам (без смешивания одной мыши в train/test) | Внедрено |
| Модели для спектров | 1D DL-подходы: RamanNet-like и Spectral Transformer-like | Внедрено |
| Dual-window стратегия | Раздельные модели для `1500` и `2900` + fusion вероятностей | Внедрено |
| Single-step preprocessing (learnable) | Проверяли как гипотезу | Протестировано, не стало основным |
| Иерархическая схема (region -> class) | Проверяли как гипотезу | Протестировано, не стало основным |
| Доп. архитектуры (Inception/DRSN/Efficient/варианты attention) | Проверяли как гипотезы | Протестировано, без устойчивого прироста |

## 2) Источники по preprocessing и валидации

1. Zhang et al. (2010), *Adaptive iteratively reweighted penalized least squares for baseline correction* (airPLS), Analyst.  
DOI: `10.1039/B922045C`  
https://pubs.rsc.org/en/content/articlehtml/2010/an/b922045c

2. Baek et al. (2015), *Asymmetrically reweighted penalized least squares smoothing* (arPLS), Analyst.  
DOI: `10.1039/C4AN01061B`  
https://pubs.rsc.org/en/content/articlehtml/2015/an/c4an01061b

3. Cordero et al. (2017), *Evaluation of SERDS and comparison to computational background correction methods*, Sensors.  
DOI: `10.3390/s17081724`  
https://pubmed.ncbi.nlm.nih.gov/28749450/

4. Chen et al. (2019), *Adaptive fully automated baseline correction based on morphological operations*, Applied Spectroscopy.  
DOI: `10.1177/0003702818811688`  
https://pubmed.ncbi.nlm.nih.gov/30334459/

5. Guo et al. (2018), *EMSC-based model transfer for Raman spectroscopy in biological applications*, Analytical Chemistry.  
DOI: `10.1021/acs.analchem.8b01536`  
https://pubmed.ncbi.nlm.nih.gov/30016081/

6. Witjes et al. (2000), *Automatic correction of peak shifts in Raman spectra before PLS regression*.  
DOI: `10.1016/S0169-7439(00)00085-X`  
https://www.sciencedirect.com/science/article/abs/pii/S016974390000085X

7. Liu & Hennelly (2024), *Wavenumber Calibration Protocol for Raman Spectrometers*.  
DOI: `10.1177/00037028241254847`  
https://pubmed.ncbi.nlm.nih.gov/38825581/

8. Savitzky & Golay (1964), *Smoothing and Differentiation of Data by Simplified Least Squares Procedures*.  
DOI: `10.1021/ac60214a047`  
https://pubs.acs.org/doi/10.1021/ac60214a047

9. Rinnan et al. (2009), *Review of the most common pre-processing techniques for near-infrared spectra*.  
DOI: `10.1016/j.trac.2009.07.007`  
https://www.sciencedirect.com/science/article/pii/S0165993609001629

10. Afseth & Kohler (2012), *Extended multiplicative signal correction in vibrational spectroscopy, a tutorial*.  
DOI: `10.1016/j.chemolab.2012.03.004`  
https://www.sciencedirect.com/science/article/pii/S0169743912000494

11. Engel et al. (2013), *Breaking with trends in pre-processing?*.  
DOI: `10.1016/j.trac.2013.04.015`  
https://www.sciencedirect.com/science/article/pii/S0165993613001465

12. Yan (2025), *A review on spectral data preprocessing techniques for machine learning and quantitative analysis*, iScience.  
DOI: `10.1016/j.isci.2025.112759`  
https://pubmed.ncbi.nlm.nih.gov/40606754/

## 3) Источники по архитектурам и ML-подходам

1. RamanNet paper (2021), segmented Raman classification architecture.  
https://link.springer.com/article/10.1007/s42452-021-04741-0

2. RamanNet reference implementation.  
https://github.com/jeff-hykin/RamanNet

3. Nature Communications paper with strong 1D DL Raman baseline (ResNet-like family).  
https://www.nature.com/articles/s41467-021-26350-w

4. Bacteria-ID reference repository (Raman DL baseline implementation).  
https://github.com/csho33/bacteria-ID

5. Single-step learnable preprocessing for Raman spectra (CNN-based).  
DOI: `10.1177/0003702819888949`  
https://pubmed.ncbi.nlm.nih.gov/31961223/

6. Quattrocchi et al. (2023), Raman + ML for grading/classification setting.  
DOI: `10.1038/s41598-023-34457-5`  
https://www.nature.com/articles/s41598-023-34457-5

7. Raman transfer learning comparison paper (1D CNN/ResNet/Inception families).  
DOI: `10.1016/j.saa.2021.120704`  
https://pubmed.ncbi.nlm.nih.gov/34509888/

8. Transformer-style spectral modeling references used при выборе направления:
https://pubmed.ncbi.nlm.nih.gov/37709483/  
https://www.sciencedirect.com/science/article/pii/S0003267023009790  
https://pubmed.ncbi.nlm.nih.gov/36067379/  
https://www.sciencedirect.com/science/article/pii/S0030401822000402

## 4) Инструменты и библиотеки, на которые опирались

1. RamanSPy (framework/docs):  
https://ramanspy.readthedocs.io/en/latest/overview.html  
https://github.com/barahona-research-group/RamanSPy

2. pybaselines (реализации baseline-коррекции):  
https://pypi.org/project/pybaselines/

## 5) Короткий итог

Итоговое решение не является «случайным набором эвристик»: оно собрано из устойчивых практик из литературы (preprocessing + корректная валидация + раздельное моделирование спектральных окон + fusion) и подтверждено серией собственных экспериментов.
