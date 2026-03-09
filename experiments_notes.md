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

### 3. Noise $\rightarrow$ Adaptive Noise

Actualmente, el ruido (`self.Sigma`) es fijo en `config.py`.

**Qué probar:**

* Si el tracker está "perdido" (el peso máximo de las partículas es bajo), **aumentar el ruido** para que las partículas se dispersen y busquen en un área más grande.
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

Es la optimización de los valores `K`, $\alpha$, `std_noise`, etc. Esto se hace al final o en paralelo probando las mejoras.

---

## Propuesta de División de Trabajo (Branches)

Para evitar conflictos de merge, os sugiero dividir el archivo `particle_filter.py` conceptualmente. Uno toca la lógica de movimiento (principio del `update`) y el otro la lógica de evaluación (final del `update`).

### Persona A: "El Navegante" (Dinámica y Estado)

* **Responsabilidad:** Controlar dónde están las partículas y cómo se mueven.
* **Archivos principales:** `particle_filter.py` (método `__init__` y primera mitad de `update`), `config.py`.
* **Tareas:**
* **Adaptive Noise:** Modificar la matriz de ruido `self.Sigma` dinámicamente en cada frame basándose en la calidad de la predicción anterior (puedes guardar el `max(self.w)` del frame anterior).
* **Modelo de Movimiento:** Implementar un "switch" en `config.py` para elegir entre `ConstantVelocity` o `RandomWalk`. Modificar la matriz `self.A` en el `__init__` según esto.



### Persona B: "El Observador" (Apariencia y Modelo)

* **Responsabilidad:** Evaluar qué tan buenas son las partículas y actualizar el modelo del objeto.
* **Archivos principales:** `particle_filter.py` (método `update` parte del bucle `for`, y funciones auxiliares), `config.py`.
* **Tareas:**
* **Modelo de Referencia Dinámico:** Al final del `update`, recalcular el histograma de la mejor partícula y mezclarlo con `self.hist_ref`.
* **Histogramas Espaciales (Opcional pero recomendado):** Modificar `computeMultiChannelHistogram` o crear una nueva función que concatene histogramas de la mitad superior e inferior de la caja.



---

## Cómo estructurar el código para el Merge

Para que el merge sea suave, **no escribáis todo el código dentro del método `update` gigante**. Refactorizad el código en métodos pequeños dentro de la clase.

Ejemplo de estructura recomendada para `particle_filter.py`:

```python
class particle_filter:
    def __init__(self, ...):
        # ... init code ...
        self.dynamic_update = False # Flag para Persona B

    # --- ZONA PERSONA A ---
    def predict_particles(self):
        # Lógica de movimiento (A @ x + noise)
        # Aquí metes la lógica de Adaptive Noise antes de aplicar el ruido
        pass

    # --- ZONA PERSONA B ---
    def compute_likelihood(self, particle_state, frame):
        # Lógica de extraer recorte, calcular histograma y Bhattacharyya
        # Aquí metes la lógica de Histogramas Espaciales
        pass

    def update_model(self, best_particle_state, frame):
        # Lógica para actualizar self.hist_ref
        pass

    # --- MÉTODO COMÚN (SOLO ORQUESTA) ---
    def update(self, im):
        # 1. Resampling (Ya hecho)
        # ...
        
        # 2. Prediction (Llamada a método de A)
        self.predict_particles() 
        
        # 3. Evaluation (Llamada a método de B dentro del bucle)
        for i in range(self.N):
             self.w[i] = self.compute_likelihood(self.x[i], im)
        
        # 4. Estimation (Ya hecho)
        # ...
        
        # 5. Model Update (Llamada a método de B)
        self.update_model(best_particle, im)

```

### Resumen de Ramas de Git

* **`main`:** Código base funcional.
* **`feature/motion-model`:** Persona A. Cambios en `__init__` (matriz A) y cálculo de `x_new` y ruido.
* **`feature/observation-model`:** Persona B. Cambios en cálculo de histogramas, `self.w` y actualización de `self.hist_ref`.

Si seguís esta estructura de métodos separados, al hacer merge, Git sabrá que uno ha añadido funciones arriba y otro abajo, y el conflicto en `update()` será mínimo (solo las llamadas a las funciones).


## Persona A: El navegante (Santi el capitan)
## Persona B: El observador. (Jorge el curioso)