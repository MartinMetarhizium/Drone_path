# 🛩️ Optimización de Ruta de Drones en Campos

Este proyecto implementa un **algoritmo evolutivo** 'worker_evolution.py' para optimizar la ruta de un dron que debe recorrer un campo (`FIELD_SIZE`) cubriéndolo por completo, minimizando la cantidad total de movimientos. El campo se divide en varias zonas (`WORKERS`), cada una de las cuales se recorre inicialmente de forma eficiente, y luego se optimiza el **orden de visita de las zonas**.

El objetivo es obtener una ruta **lo más corta posible** en comparación con el mínimo teórico.

---

## 📦 Requisitos

- Python 3.7+
- Bibliotecas:
  - `numpy`
  - `matplotlib`


