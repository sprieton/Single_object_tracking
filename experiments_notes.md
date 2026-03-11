## Explicación de las Tareas

### 1. $p(z_k|x_k^{i})$ (Modelo de Observación)

Esto define cómo de "parecida" es una partícula al objeto real. Actualmente, usas un histograma de color HS global y la distancia de Bhattacharyya.

**Qué probar:**

* **Histogramas Espaciales:** En lugar de un histograma para toda la caja, divide la caja en 2 partes (arriba/abajo) o 4 cuadrantes. Esto ayuda a que el tracker sepa que la "cabeza" debe estar arriba y los "pies" abajo, evitando que el tracker se centre en una mancha de color similar en el fondo.
* **Otras métricas:** Probar Chi-Square o Intersección de histogramas en lugar de Bhattacharyya.
* **Espacios de color:** Probar Lab o RGB en vez de HSV.

### 2. $p(x_k^{(i)}|x_{k-i}^{(i)})$ (Modelo de Movimiento / Transición)

Esto define cómo se mueven las partículas de un frame a otro. Actualmente, usas un modelo de Velocidad Constante.

**Qué probar:**

* **Random Walk (Posición Constante):** Asumir que la velocidad es 0 y el objeto se mueve aleatoriamente por el ruido. A veces funciona mejor si el objeto cambia de dirección muy bruscamente (como en el video de *Basketball*).
* **Aceleración:** Incluir aceleración en el estado (vector de 12 dimensiones).
* **Reinicio de partículas perdidas:**  
  Cuando el tracker se pierde (Neff bajo), se puede dispersar un porcentaje de partículas alrededor de la última posición conocida o incluso por toda la imagen con distribución gaussiana. Esto permite que algunas partículas “exploren” zonas lejanas donde podría estar el objeto.  
  > **Nota:** Solo con reinicio aleatorio las partículas no se centran correctamente, y el tracker puede tardar en volver a encontrar el objeto si el modelo de observación no discrimina bien. Mejorar el modelo de observación aumenta mucho la efectividad de esta técnica.

### 3. Noise $\rightarrow$ Adaptive Noise

Actualmente, el ruido (`self.Sigma`) es fijo en `config.py`.

**Qué probar:**

* Si el tracker está "perdido" (Neff bajo), **aumentar el ruido** para que las partículas se dispersen y busquen en un área más grande.
* Si el tracker está "seguro" (peso alto), **reducir el ruido** para concentrarse y ser más preciso.
* Ajustar el ruido proporcionalmente a la velocidad del objeto (si se mueve rápido, mayor incertidumbre).

### 4. Ref model $\rightarrow$ Fixed vs Update

Actualmente, `self.hist_ref` se calcula en el frame 1 y nunca cambia.

**Qué probar:**

* **Actualización lineal:** 
$$H_{ref}^{t} = (1 - \alpha) \cdot H_{ref}^{t-1} + \alpha \cdot H_{best\_particle}$$

Esto permite adaptarse a cambios de iluminación o rotación del objeto.

> **Peligro:** Si actualizas demasiado rápido, el tracker puede aprenderse el fondo y derivar (drifting). Hay que hacerlo con cuidado (un $\alpha$ muy pequeño, ej. 0.01).

### 5. Parameters (`config.py`)

Optimización de valores `K`, $\alpha$, `std_noise`, `lost_obj_part_restart`, etc. Esto se hace al final o en paralelo probando las mejoras.

---

## Observaciones sobre reinicio de partículas

Cuando `Neff` cae por debajo de `cfg.lost_obj_Neff_th`, un porcentaje de partículas (`cfg.lost_obj_part_restart`) se puede reiniciar con distribución gaussiana alrededor de la última posición conocida del objeto:

```python
H, W = im_shape[:2]
x_global = self.x.mean(axis=0)  # última posición conocida
for idx in range(num_reset):
    self.x[idx, 0] = np.clip(x_global[0] + npr.randn() * 0.5 * W, 0, W)
    self.x[idx, 1] = np.clip(x_global[1] + npr.randn() * 0.5 * H, 0, H)
    self.x[idx, 2] = np.clip(x_global[2] + npr.randn() * 0.5 * x_global[2], 1, W)
    self.x[idx, 3] = np.clip(x_global[3] + npr.randn() * 0.5 * x_global[3], 1, H)
    self.x[idx, 4:] = 0
```

# Tracker de resultados:

Mejor hasta ahora:

Base de Santi: Total average Results are JI=0.516555

Histograma partido: Total average Results are JI=0.515722

Vale ahora genera el plan para implementar que el objeto de referencia se vaya actualizando. 