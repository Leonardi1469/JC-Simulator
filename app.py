# ============================================================
# JC SIMULATOR
# ============================================================
#
# Simulador interactivo del modelo de Jaynes-Cummings
#
# Tesis de Licenciatura
# Ingeniería en Desarrollo y Tecnologías de Software
#
# Autores:
# Leonardi Hernández Sánchez
# Emily Andrea Franco Escudero
#
# 2026
# ============================================================


# ============================================================
# 1. IMPORTACIÓN DE BIBLIOTECAS
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO

from qutip import (
    basis,
    coherent,
    destroy,
    displace,
    entropy_vn,
    expect,
    fock,
    ket2dm,
    qeye,
    sesolve,
    sigmam,
    sigmap,
    sigmaz,
    squeeze,
    tensor,
)


# ============================================================
# 2. CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================

st.set_page_config(
    page_title="JC Simulator",
    layout="wide",
)

st.title("Simulador del Modelo de Jaynes-Cummings")

st.caption(
    "JC Simulator v1.0 • "
    "Developed by Leonardi Hernández Sánchez and "
    "Emily Andrea Franco Escudero • 2026"
)

with st.expander("Descripción del modelo físico"):

    st.markdown(
        """
El presente simulador resuelve la ecuación de Schrödinger para el
modelo de Jaynes-Cummings, el cual describe la interacción entre un
átomo de dos niveles y un único modo cuantizado del campo
electromagnético dentro de una cavidad.
        """
    )

    st.markdown(
        r"""
El Hamiltoniano del sistema, en unidades donde
$\hbar=1$, está dado por:
        """
    )

    st.latex(
        r"""
        \hat{H}
        =
        \omega_c \hat{a}^{\dagger}\hat{a}
        +
        \frac{\omega_{eg}}{2}\hat{\sigma}_z
        +
        g\left(
        \hat{a}\hat{\sigma}_{+}
        +
        \hat{a}^{\dagger}\hat{\sigma}_{-}
        \right)
        """
    )

    st.markdown("donde:")

    st.markdown(
        r"""
- $\omega_c$: frecuencia angular del modo de la cavidad.

- $\omega_{eg}$: frecuencia de transición entre los estados atómicos.

- $g$: constante de acoplamiento entre el átomo y el campo.

- $\hat{a}^{\dagger}$ y $\hat{a}$: operadores de creación y
  destrucción del campo cuantizado.

- $\hat{\sigma}_z$, $\hat{\sigma}_{+}$ y
  $\hat{\sigma}_{-}$: operadores del sistema atómico de dos niveles.
        """
    )

    st.info(
        "El simulador permite estudiar la dinámica temporal del sistema "
        "para diversos estados iniciales del átomo y del campo. También "
        "permite analizar las probabilidades atómicas, la inversión "
        "atómica, el número promedio de fotones, la entropía de "
        "von Neumann y el número total de excitaciones."
    )


# ============================================================
# 3. DEFINICIÓN DE LOS PARÁMETROS DE ENTRADA
# ============================================================

col_izq, col_der = st.columns([1, 2])


with col_izq:

    # --------------------------------------------------------
    # 3.1 Parámetros físicos del modelo
    # --------------------------------------------------------

    st.header("Parámetros del modelo")

    wc = st.number_input(
        "Frecuencia de la cavidad $\\omega_c$",
        min_value=0.0,
        value=1.0,
        step=0.01,
        format="%.3f",
    )

    wa = st.number_input(
        "Frecuencia atómica $\\omega_{eg}$",
        min_value=0.0,
        value=1.0,
        step=0.01,
        format="%.3f",
    )

    g = st.number_input(
        "Constante de acoplamiento $g$",
        min_value=0.0,
        value=1.0,
        step=0.01,
        format="%.3f",
    )

    N = st.slider(
        "Dimensión del espacio de Fock $N$",
        min_value=2,
        max_value=100,
        value=20,
        step=1,
    )

    tmax = st.number_input(
        "Tiempo máximo $t_{\\max}$",
        min_value=0.1,
        value=20.0,
        step=0.1,
        format="%.2f",
    )

    Nt = st.number_input(
        "Número de puntos temporales $N_t$",
        min_value=100,
        value=1000,
        step=100,
    )

    # --------------------------------------------------------
    # 3.2 Estado inicial del átomo
    # --------------------------------------------------------

    st.header("Estado inicial del átomo")

    estado_atomo = st.radio(
        "Seleccione el estado atómico",
        options=[
            "Excitado",
            "Base",
            "Superposición",
        ],
    )

    theta = 0.0
    phi = 0.0

    if estado_atomo == "Superposición":

        theta = st.slider(
            "Ángulo polar $\\theta$",
            min_value=0.0,
            max_value=float(np.pi),
            value=float(np.pi / 2),
        )

        st.latex(
            rf"\theta={theta / np.pi:.3f}\,\pi"
        )

        phi = st.slider(
            "Ángulo azimutal $\\phi$",
            min_value=0.0,
            max_value=float(2 * np.pi),
            value=0.0,
        )

        st.latex(
            rf"\phi={phi / np.pi:.3f}\,\pi"
        )

        st.markdown("Estado atómico seleccionado:")

        st.latex(
            rf"""
            |\psi_a(0)\rangle
            =
            {np.cos(theta / 2):.3f}|e\rangle
            +
            e^{{i({phi:.3f})}}
            {np.sin(theta / 2):.3f}|g\rangle
            """
        )

    # --------------------------------------------------------
    # 3.3 Estado inicial del campo
    # --------------------------------------------------------

    st.header("Estado inicial del campo")

    estado_campo = st.radio(
        "Seleccione el estado del campo",
        options=[
            "Vacío",
            "Fock",
            "Coherente",
            "Comprimido",
            "Coherente comprimido",
        ],
    )

    n0 = 0
    alpha = 2.0
    r = 0.5

    if estado_campo == "Fock":

        n0 = st.number_input(
            "Número de fotones $n$",
            min_value=0,
            max_value=N - 1,
            value=1,
            step=1,
        )

    elif estado_campo == "Coherente":

        alpha = st.number_input(
            "Amplitud coherente $\\alpha$",
            value=2.0,
            step=0.1,
            format="%.3f",
        )

    elif estado_campo == "Comprimido":

        r = st.number_input(
            "Parámetro de compresión $r$",
            min_value=0.0,
            value=0.5,
            step=0.1,
            format="%.3f",
        )

    elif estado_campo == "Coherente comprimido":

        alpha = st.number_input(
            "Amplitud coherente $\\alpha$",
            value=2.0,
            step=0.1,
            format="%.3f",
        )

        r = st.number_input(
            "Parámetro de compresión $r$",
            min_value=0.0,
            value=0.5,
            step=0.1,
            format="%.3f",
        )

    # --------------------------------------------------------
    # 3.4 Botón de ejecución
    # --------------------------------------------------------

    ejecutar = st.button(
        "Ejecutar simulación",
        type="primary",
        use_container_width=True,
    )


with col_der:

    if ejecutar:

        # ====================================================
        # 4. CONSTRUCCIÓN DEL MODELO CUÁNTICO
        # ====================================================

        # ----------------------------------------------------
        # 4.1 Construcción del vector temporal
        # ----------------------------------------------------

        t = np.linspace(
            0.0,
            float(tmax),
            int(Nt),
        )

        # ----------------------------------------------------
        # 4.2 Construcción de los operadores cuánticos
        # ----------------------------------------------------

        a = tensor(
            destroy(N),
            qeye(2),
        )

        sz = tensor(
            qeye(N),
            sigmaz(),
        )

        sp = tensor(
            qeye(N),
            sigmap(),
        )

        sm = tensor(
            qeye(N),
            sigmam(),
        )

        # ----------------------------------------------------
        # 4.3 Construcción del Hamiltoniano
        # ----------------------------------------------------

        H = (
            wc * a.dag() * a
            + 0.5 * wa * sz
            + g * (
                a * sp
                + a.dag() * sm
            )
        )

        # ----------------------------------------------------
        # 4.4 Construcción del estado inicial del campo
        # ----------------------------------------------------

        if estado_campo == "Vacío":

            psi_f = fock(N, 0)

        elif estado_campo == "Fock":

            psi_f = fock(
                N,
                int(n0),
            )

        elif estado_campo == "Coherente":

            psi_f = coherent(
                N,
                alpha,
            )

        elif estado_campo == "Comprimido":

            psi_f = (
                squeeze(N, r)
                * fock(N, 0)
            )

        elif estado_campo == "Coherente comprimido":

            psi_f = (
                displace(N, alpha)
                * squeeze(N, r)
                * fock(N, 0)
            )

        else:

            psi_f = fock(N, 0)

        # ----------------------------------------------------
        # 4.5 Construcción del estado inicial del átomo
        # ----------------------------------------------------

        if estado_atomo == "Excitado":

            psi_a = basis(2, 0)

        elif estado_atomo == "Base":

            psi_a = basis(2, 1)

        elif estado_atomo == "Superposición":

            psi_a = (
                np.cos(theta / 2)
                * basis(2, 0)
                + np.exp(1j * phi)
                * np.sin(theta / 2)
                * basis(2, 1)
            )

        else:

            psi_a = basis(2, 0)

        # ----------------------------------------------------
        # 4.6 Construcción del estado inicial total
        # ----------------------------------------------------

        psi0 = tensor(
            psi_f,
            psi_a,
        )


        # ====================================================
        # 5. EVOLUCIÓN TEMPORAL DEL SISTEMA
        # ====================================================

        # ----------------------------------------------------
        # 5.1 Resolución de la ecuación de Schrödinger
        # ----------------------------------------------------

        with st.spinner(
            "Calculando la evolución temporal del sistema..."
        ):

            resultado = sesolve(
                H,
                psi0,
                t,
            )

            estados = resultado.states


        # ====================================================
        # 6. CÁLCULO DE OBSERVABLES FÍSICOS
        # ====================================================

        # ----------------------------------------------------
        # 6.1 Definición de los operadores observables
        # ----------------------------------------------------

        proyector_e = (
            basis(2, 0)
            * basis(2, 0).dag()
        )

        proyector_g = (
            basis(2, 1)
            * basis(2, 1).dag()
        )

        Pe_op = tensor(
            qeye(N),
            proyector_e,
        )

        Pg_op = tensor(
            qeye(N),
            proyector_g,
        )

        n_op = a.dag() * a

        M_op = n_op + Pe_op

        # ----------------------------------------------------
        # 6.2 Cálculo de las probabilidades atómicas
        # ----------------------------------------------------

        Pe = np.real(
            np.asarray(
                expect(Pe_op, estados),
                dtype=complex,
            )
        )

        Pg = np.real(
            np.asarray(
                expect(Pg_op, estados),
                dtype=complex,
            )
        )

        # ----------------------------------------------------
        # 6.3 Cálculo de la inversión atómica
        # ----------------------------------------------------

        W = Pe - Pg

        # ----------------------------------------------------
        # 6.4 Cálculo del número promedio de fotones
        # ----------------------------------------------------

        nmean = np.real(
            np.asarray(
                expect(n_op, estados),
                dtype=complex,
            )
        )

        # ----------------------------------------------------
        # 6.5 Cálculo del número total de excitaciones
        # ----------------------------------------------------

        Mmean = np.real(
            np.asarray(
                expect(M_op, estados),
                dtype=complex,
            )
        )

        # ----------------------------------------------------
        # 6.6 Cálculo de la entropía de von Neumann
        # ----------------------------------------------------

        S = []

        for psi_t in estados:

            rho = ket2dm(psi_t)

            rho_atomo = rho.ptrace(1)

            S.append(
                entropy_vn(rho_atomo)
            )

        S = np.real(
            np.asarray(S)
        )


        # ====================================================
        # 7. VISUALIZACIÓN DE RESULTADOS
        # ====================================================

        # ----------------------------------------------------
        # 7.1 Construcción de la figura
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            3,
            2,
            figsize=(8, 10),
            dpi=300,
        )

        # ----------------------------------------------------
        # 7.2 Probabilidad del estado excitado
        # ----------------------------------------------------

        ax[0, 0].plot(
            t,
            Pe,
        )

        ax[0, 0].set(
            xlabel=r"$t$",
            ylabel=r"$P_e(t)$",
            title="Probabilidad del estado excitado",
            ylim=(0, 1),
        )

        # ----------------------------------------------------
        # 7.3 Probabilidad del estado base
        # ----------------------------------------------------

        ax[0, 1].plot(
            t,
            Pg,
        )

        ax[0, 1].set(
            xlabel=r"$t$",
            ylabel=r"$P_g(t)$",
            title="Probabilidad del estado base",
            ylim=(0, 1),
        )

        # ----------------------------------------------------
        # 7.4 Inversión atómica
        # ----------------------------------------------------

        ax[1, 0].plot(
            t,
            W,
        )

        ax[1, 0].set(
            xlabel=r"$t$",
            ylabel=r"$W(t)$",
            title="Inversión atómica",
            ylim=(-1, 1),
        )

        ax[1, 0].set_yticks(
            np.arange(
                -1,
                1.1,
                0.5,
            )
        )

        # ----------------------------------------------------
        # 7.5 Número promedio de fotones
        # ----------------------------------------------------

        ax[1, 1].plot(
            t,
            nmean,
        )

        ax[1, 1].set(
            xlabel=r"$t$",
            ylabel=r"$\langle n(t)\rangle$",
            title="Número promedio de fotones",
        )

        # ----------------------------------------------------
        # 7.6 Entropía de von Neumann
        # ----------------------------------------------------

        ax[2, 0].plot(
            t,
            S,
        )

        ax[2, 0].set(
            xlabel=r"$t$",
            ylabel=r"$S(t)$",
            title="Entropía de von Neumann",
            ylim=(0, 1),
        )

        # ----------------------------------------------------
        # 7.7 Número total de excitaciones
        # ----------------------------------------------------

        valor_M = float(
            np.mean(Mmean)
        )

        margen_M = max(
            0.5,
            float(
                np.max(
                    np.abs(
                        Mmean - valor_M
                    )
                )
            )
            + 0.1,
        )

        ax[2, 1].plot(
            t,
            Mmean,
        )

        ax[2, 1].set(
            xlabel=r"$t$",
            ylabel=r"$\langle M(t)\rangle$",
            title="Número total de excitaciones",
            ylim=(
                valor_M - margen_M,
                valor_M + margen_M,
            ),
        )

        # ----------------------------------------------------
        # 7.8 Formato general de las gráficas
        # ----------------------------------------------------

        for fila in ax:

            for eje in fila:

                eje.tick_params(
                    axis="both",
                    labelsize=9,
                )

                eje.xaxis.label.set_size(11)
                eje.yaxis.label.set_size(11)
                eje.title.set_size(12)

                eje.grid(
                    True,
                    alpha=0.5,
                )

                eje.spines[
                    "top"
                ].set_visible(False)

                eje.spines[
                    "right"
                ].set_visible(False)

        fig.suptitle(
            "Simulación del modelo de Jaynes-Cummings",
            fontsize=16,
        )

        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.98,
            ]
        )

        # ----------------------------------------------------
        # 7.9 Presentación de la figura en la interfaz
        # ----------------------------------------------------

        st.pyplot(
            fig,
            use_container_width=True,
        )

        st.success(
            "La simulación se realizó correctamente."
        )


        # ====================================================
        # 8. EXPORTACIÓN DE RESULTADOS
        # ====================================================

        # ----------------------------------------------------
        # 8.1 Creación de los archivos en memoria
        # ----------------------------------------------------

        pdf_buffer = BytesIO()
        png_buffer = BytesIO()

        fig.savefig(
            pdf_buffer,
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            png_buffer,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

        pdf_buffer.seek(0)
        png_buffer.seek(0)

        # ----------------------------------------------------
        # 8.2 Presentación de las opciones de descarga
        # ----------------------------------------------------

        st.subheader(
            "Exportar resultados"
        )

        col_pdf, col_png = st.columns(2)

        # ----------------------------------------------------
        # 8.3 Descarga de la figura en formato PDF
        # ----------------------------------------------------

        with col_pdf:

            st.download_button(
                label="Descargar figura en PDF",
                data=pdf_buffer.getvalue(),
                file_name="Resultados_JC.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # ----------------------------------------------------
        # 8.4 Descarga de la figura en formato PNG
        # ----------------------------------------------------

        with col_png:

            st.download_button(
                label="Descargar figura en PNG",
                data=png_buffer.getvalue(),
                file_name="Resultados_JC.png",
                mime="image/png",
                use_container_width=True,
            )

        plt.close(fig)

    else:

        st.info(
            "Configure los parámetros y presione "
            "**Ejecutar simulación**."
        )
