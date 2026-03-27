import streamlit as st
import io
import zipfile
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import datetime
import warnings
import pandas as pd
import numpy as np

# Configurar pandas para que no muestre advertencias de formato de fecha
pd.options.mode.chained_assignment = None  # default='warn'

# Suprimir advertencias específicas
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Parsing dates in.*%d/%m/%Y.*"
)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------------------
# Configuración inicial
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Sugeridor de Materiales - Simple", layout="wide")
st.title("📊 Sugeridor de Materiales - Asignación 1:1")


# ── Utilidad: medidor de tiempo para UI ──────────────────────────────────────
class Timer:
    """Cronómetro ligero para mostrar tiempos en la UI."""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> str:
        s = time.perf_counter() - self._start
        return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ------------------------------------------------------------------------------
# Definición de columnas (en el orden solicitado) - SIN SIMILITUD
# ------------------------------------------------------------------------------
class Columnas:
    GRUPO_CLIENTE = "Gpo. Cte."
    FECHA = "Fecha"
    PEDIDO = "Pedido"
    GRUPO_VENDEDOR = "Gpo.Vdor."
    SOLICITANTE = "Solicitante"
    DESTINATARIO = "Destinatario"
    RAZON_SOCIAL = "Razón Social"
    CENTRO_PEDIDO = "Centro pedido"
    ALMACEN = "Almacén"
    MATERIAL_SOLICITADO = "Material solicitado"
    MATERIAL_BASE = "Material base"
    DESCRIPCION_SOLICITADA = "Descripción solicitada"
    CANTIDAD_PEDIDO = "Cantidad pedido"
    CANTIDAD_PENDIENTE = "Cantidad pendiente"
    CANTIDAD_OFERTAR = "Cantidad a Ofertar"
    PRECIO = "Precio"
    FUENTE = "Fuente"
    MATERIAL_SUGERIDO = "Material sugerido"  # NUEVA COLUMNA
    DESCRIPCION_SUGERIDA = "Descripción sugerida"  # NUEVA COLUMNA
    CENTRO_SUGERIDO = "Centro sugerido"
    ALMACEN_SUGERIDO = "Almacén sugerido"
    DISPONIBLE = "Disponible"
    LOTE = "Lote"
    FECHA_CADUCIDAD = "Fecha de Caducidad"
    # SIMILITUD eliminado según solicitud
    CENTRO_INV = "Centro (Inv)"
    INV_1030 = "Inv 1030"
    INV_1031 = "Inv 1031"
    INV_1032 = "Inv 1032"
    INV_1060 = "Inv 1060"
    MESES_INVENTARIO = "Meses_Inventario"
    PROMEDIO_CONSUMO_12M = "Promedio_Consumo_12M"
    CONSUMO_DESTINATARIO_12M = "Consumo promedio (Destinatario/Material)"
    CANT_TRANSITO = "Cant. en Tránsito"
    CANT_TRANSITO_1030 = "Cant. en Tránsito 1030"  # NUEVA
    CANT_TRANSITO_1031 = "Cant. en Tránsito 1031"  # NUEVA
    CANT_TRANSITO_1032 = "Cant. en Tránsito 1032"  # NUEVA
    DISP_1031_1030 = "Disponible 1031-1030"
    DISP_1031_1032 = "Disponible 1031-1032"
    INV_1001 = "Inv 1001"
    INV_1003 = "Inv 1003"
    INV_1004 = "Inv 1004"
    INV_1017 = "Inv 1017"
    INV_1018 = "Inv 1018"
    INV_1022 = "Inv 1022"
    INV_1036 = "Inv 1036"
    BLOQUEADO = "Bloqueado"


# ------------------------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------------------------
def normalizar_ids(serie: pd.Series) -> pd.Series:
    """Normaliza IDs quitando espacios y sufijos .0"""
    # Si es un string vacío, devolver serie vacía
    if isinstance(serie, str):
        return pd.Series([], dtype=str)

    # Si es una serie, procesarla normalmente
    return serie.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def encontrar_columna_por_patron(
    df: pd.DataFrame, patrones: List[str]
) -> Optional[str]:
    """Busca una columna que coincida con alguno de los patrones (case insensitive)."""
    for col in df.columns:
        col_lower = col.lower()
        for patron in patrones:
            if patron.lower() in col_lower:
                return col
    return None


def formatear_fecha_caducidad(fecha) -> str:
    """Normaliza fechas a dd/mm/aaaa sin reinterpretar formatos ya correctos."""
    if pd.isna(fecha):
        return ""

    if isinstance(fecha, str):
        fecha = fecha.strip()
        if not fecha or fecha.lower() == "nan":
            return ""

    try:
        fecha_dt = pd.to_datetime(fecha, dayfirst=True, errors="coerce")
        if pd.notnull(fecha_dt):
            return fecha_dt.strftime("%d/%m/%Y")
    except Exception:
        pass

    return str(fecha).strip()


def procesar_hoja_inventario_ajustada(df_inventario: pd.DataFrame) -> pd.DataFrame:
    """Procesa la hoja de inventario y realiza el cálculo: 'Libre Utilización' - 'Entrega a cliente'"""
    if df_inventario.empty:
        return pd.DataFrame()

    # Normalizar nombres de columnas
    df_inventario.columns = [
        col.replace("Almacen", "Almacén").replace("Almaçen", "Almacén")
        for col in df_inventario.columns
    ]

    # Buscar columnas por patrones (AGREGAR "Entrega a cliente" y "Descripción")
    columnas_requeridas = [
        "Centro",
        "Material",
        "Almacén",
        "Libre Utilización",
        "Cant. en Tránsito",
        "Entrega a cliente",  # NUEVA COLUMNA REQUERIDA
        "Descripción",  # NUEVA: Columna para descripción del material
    ]

    mapeo_columnas = {}

    for col_req in columnas_requeridas:
        if col_req not in df_inventario.columns:
            patrones = {
                "Centro": ["centro", "center"],
                "Material": ["material", "mat", "artículo"],
                "Almacén": ["almacén", "almacen", "almacen"],
                "Libre Utilización": [
                    "libre utilización",
                    "libre utilizacion",
                    "disponible",
                    "stock",
                ],
                "Cant. en Tránsito": [
                    "tránsito",
                    "transito",
                    "en tránsito",
                    "en transito",
                    "cant. en tránsito",
                ],
                "Entrega a cliente": [  # NUEVOS PATRONES
                    "entrega a cliente",
                    "entrega cliente",
                    "entregado",
                    "cantidad entregada",
                    "entregas",
                ],
                "Descripción": [  # NUEVO: Patrones para buscar descripción
                    "descripción",
                    "descripcion",
                    "texto breve",
                    "texto material",
                    "nombre",
                    "texto",
                    "descr",
                    "artículo",
                ],
            }
            col_encontrada = encontrar_columna_por_patron(
                df_inventario, patrones.get(col_req, [col_req])
            )
            if col_encontrada:
                mapeo_columnas[col_req] = col_encontrada
            else:
                # Si no se encuentra, crear columna con valor vacío para texto o 0 para numéricas
                if col_req in [
                    "Libre Utilización",
                    "Cant. en Tránsito",
                    "Entrega a cliente",
                ]:
                    df_inventario[col_req] = 0
                else:
                    df_inventario[col_req] = ""

    # Renombrar columnas según mapeo
    for col_dest, col_orig in mapeo_columnas.items():
        if col_orig in df_inventario.columns and col_dest not in df_inventario.columns:
            df_inventario[col_dest] = df_inventario[col_orig]

    # Normalizar IDs
    for col in ["Centro", "Material", "Almacén"]:
        if col in df_inventario.columns:
            df_inventario[col] = normalizar_ids(df_inventario[col])

    # Convertir numéricos (AGREGAR "Entrega a cliente")
    columnas_numericas = ["Libre Utilización", "Cant. en Tránsito", "Entrega a cliente"]

    for col in columnas_numericas:
        if col in df_inventario.columns:
            df_inventario[col] = pd.to_numeric(
                df_inventario[col], errors="coerce"
            ).fillna(0)

    # ------------------------------------------------------------------
    # REALIZAR EL CÁLCULO SOLICITADO: "Libre Utilización" - "Entrega a cliente"
    # ------------------------------------------------------------------
    if (
        "Libre Utilización" in df_inventario.columns
        and "Entrega a cliente" in df_inventario.columns
    ):
        st.info(
            "⚠️ **Cálculo aplicado:** Se ha ajustado el inventario restando 'Entrega a cliente' de 'Libre Utilización'"
        )

        # Calcular el nuevo valor de Libre Utilización
        df_inventario["Libre Utilización"] = (
            df_inventario["Libre Utilización"] - df_inventario["Entrega a cliente"]
        )

        # Asegurar que no haya valores negativos
        df_inventario["Libre Utilización"] = df_inventario["Libre Utilización"].clip(
            lower=0
        )

        # Mostrar estadísticas del ajuste
        total_ajuste = (df_inventario["Entrega a cliente"]).sum()
        st.sidebar.write(
            f"**Ajuste aplicado:** {total_ajuste:,.0f} unidades restadas del inventario"
        )

    # Mantener columnas relevantes (AGREGAR "Descripción")
    columnas_finales = [
        "Centro",
        "Material",
        "Almacén",
        "Descripción",  # NUEVA: Agregar descripción
        "Libre Utilización",
        "Cant. en Tránsito",
    ]
    columnas_finales = [col for col in columnas_finales if col in df_inventario.columns]

    return df_inventario[columnas_finales]


# ------------------------------------------------------------------------------
# Función para limpiar cache
# ------------------------------------------------------------------------------
def limpiar_cache():
    """Limpia todos los datos cacheados"""
    if "cache_inicializado" in st.session_state:
        st.session_state.cache_pedidos = None
        st.session_state.cache_inventario = None
        st.session_state.cache_externas = None
        st.session_state.cache_facturacion = None
    st.success("Cache limpiado exitosamente")


# ------------------------------------------------------------------------------
# MODIFICAR: procesar_hoja_externa para normalizar mejor las columnas
# ------------------------------------------------------------------------------
def procesar_hoja_externa(df_externo: pd.DataFrame, nombre_hoja: str) -> pd.DataFrame:
    """Procesa hojas externas (Corta caducidad, Lento mov, etc.)"""
    if df_externo.empty:
        return pd.DataFrame()

    # Normalizar nombres de columnas
    df_externo.columns = [
        col.replace("Almacen", "Almacén").replace("Almaçen", "Almacén")
        for col in df_externo.columns
    ]

    # Agregar nombre de la hoja como atributo
    df_externo.attrs["nombre_hoja"] = nombre_hoja

    # Columnas base requeridas
    columnas_base = ["Material", "Centro", "Almacén", "CantidadDisp"]

    # Para cada hoja, buscar columnas por patrones
    columnas_a_buscar = {}

    if nombre_hoja == "Corta caducidad":
        columnas_a_buscar = {
            "Material": ["material", "mat", "artículo"],
            "Centro": ["centro", "center"],
            "Almacén": ["almacén", "almacen"],
            "CantidadDisp": [
                "cantidad",
                "disp",
                "disponible",
                "stock",
                "libre utilización",
                "libre utilizacion",
            ],
            "Descripcion": ["descripción", "descripcion", "desc", "texto"],
            "Lote": ["lote", "batch", "lote"],
            "FechaCaducidad": [
                "caducidad",
                "fecha caducidad",
                "vencimiento",
                "expira",
                "fecaduc/feprefercons",
            ],
        }
    elif nombre_hoja == "Lento mov":
        columnas_a_buscar = {
            "Material": ["material", "mat", "artículo"],
            "Descripcion": [
                "descripción",
                "descripcion",
                "desc",
                "texto",
                "texto breve",
            ],
        }
    elif nombre_hoja == "Cosmopark":
        columnas_a_buscar = {
            "Material": ["material", "mat", "artículo", "codigo"],
            "Centro": ["centro", "center"],
            "CantidadDisp": ["cantidad", "disp", "disponible", "stock"],
            "Descripcion": [
                "descripción",
                "descripcion",
                "desc",
                "texto",
                "texto material",
            ],
            "Lote": ["lote", "batch", "lote"],
            "FechaCaducidad": ["caducidad", "fecha caducidad", "vencimiento", "expira"],
        }
    elif nombre_hoja == "Sustituto":
        columnas_a_buscar = {
            "Material": ["material", "mat", "artículo"],
            "Material sustituto": ["material sustituto", "sustituto", "alternativo"],
            "Texto material sustituto": [
                "texto material sustituto",
                "descripción sustituto",
                "desc sustituto",
            ],
        }
    elif nombre_hoja in ["PNC", "Caduco"]:
        columnas_a_buscar = {
            "Material": ["material", "mat", "artículo"],
            "Centro": ["centro", "center"],
            "Almacén": ["almacén", "almacen"],
            "CantidadDisp": ["cantidad", "disp", "disponible", "stock"],
            "Descripcion": ["descripción", "descripcion", "desc", "texto"],
            "Lote": ["lote", "batch", "lote"],
            "FechaCaducidad": ["caducidad", "fecha caducidad", "vencimiento", "expira"],
        }
    else:
        columnas_a_buscar = {}

    # Buscar y asignar columnas
    mapeo_encontrado = {}
    for col_std, patrones in columnas_a_buscar.items():
        col_encontrada = encontrar_columna_por_patron(df_externo, patrones)
        if col_encontrada:
            mapeo_encontrado[col_std] = col_encontrada
        elif col_std == "Material":
            # Buscar material en cualquier columna numérica que pueda ser ID
            for col in df_externo.columns:
                if (
                    df_externo[col].dtype in ["int64", "float64"]
                    and df_externo[col].astype(str).str.match(r"^\d+$").any()
                ):
                    mapeo_encontrado[col_std] = col
                    break

    # Renombrar columnas según mapeo encontrado
    for col_std, col_orig in mapeo_encontrado.items():
        if col_orig in df_externo.columns and col_std not in df_externo.columns:
            df_externo[col_std] = df_externo[col_orig]

    # Asegurar columnas requeridas
    for col in columnas_base:
        if col not in df_externo.columns:
            df_externo[col] = 0 if col == "CantidadDisp" else ""

    # Normalizar IDs
    for col in ["Centro", "Material", "Almacén"]:
        if col in df_externo.columns:
            df_externo[col] = normalizar_ids(df_externo[col])

    # Convertir cantidades a numérico - IMPORTANTE: manejar diferentes formatos
    if "CantidadDisp" in df_externo.columns:
        df_externo["CantidadDisp"] = pd.to_numeric(
            df_externo["CantidadDisp"], errors="coerce"
        ).fillna(0)

        # Si todos los valores son 0, intentar buscar otra columna de cantidad
        if df_externo["CantidadDisp"].sum() == 0 and nombre_hoja in [
            "Cosmopark",
            "PNC",
        ]:
            for col in df_externo.columns:
                if any(term in col.lower() for term in ["cant", "qty", "quantity"]):
                    df_externo["CantidadDisp"] = pd.to_numeric(
                        df_externo[col], errors="coerce"
                    ).fillna(0)
                    break

    # Procesar fecha de caducidad
    if "FechaCaducidad" in df_externo.columns:
        df_externo["FechaCaducidad"] = pd.to_datetime(
            df_externo["FechaCaducidad"],
            dayfirst=True,
            errors="coerce",  # Agregar dayfirst=True
        )
        # Formatear a dd/mm/aaaa
        df_externo["FechaCaducidad"] = df_externo["FechaCaducidad"].apply(
            lambda x: x.strftime("%d/%m/%Y") if pd.notnull(x) else ""
        )

    # DEPURACIÓN: Mostrar columnas encontradas
    if nombre_hoja == "Lento mov":
        logger.info(f"Columnas en hoja 'Lento mov': {df_externo.columns.tolist()}")
        if "Material" in df_externo.columns:
            logger.info(
                f"Materiales en 'Lento mov': {df_externo['Material'].head().tolist()}"
            )

    return df_externo


def calcular_estadisticas_facturacion_por_almacen(
    df_facturacion: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula estadísticas de facturación por Centro/Almacén/Material:
    1. Última fecha de facturación (mm/aaaa) y suma total de ese mes
    2. Penúltima fecha de facturación (mm/aaaa) y suma total de ese mes
    3. Suma de cantidad facturada por mes
    """
    if df_facturacion.empty:
        return pd.DataFrame()

    try:
        # Asegurar que tenemos las columnas necesarias
        columnas_necesarias = [
            "Centro",
            "Material",
            "Almacén",
            "Fecha",
            "Cantidad",
            "Importe",
        ]
        for col in columnas_necesarias:
            if col not in df_facturacion.columns:
                st.warning(f"Columna {col} no encontrada en datos de facturación")
                return pd.DataFrame()

        # Convertir fecha a datetime
        df_facturacion["Fecha"] = pd.to_datetime(
            df_facturacion["Fecha"], errors="coerce"
        )

        # Crear columna de mes-año (mm/aaaa)
        df_facturacion["MesAno"] = df_facturacion["Fecha"].dt.strftime("%m/%Y")

        # Filtrar solo datos válidos (fecha válida; cantidades negativas = devoluciones son válidas)
        df_valido = df_facturacion[(df_facturacion["Fecha"].notna())].copy()

        if df_valido.empty:
            return pd.DataFrame()

        # Agrupar por Centro, Almacén, Material y MesAno
        df_agrupado = (
            df_valido.groupby(["Centro", "Almacén", "Material", "MesAno"])
            .agg(
                {
                    "Cantidad": "sum",
                    "Importe": "sum",
                    "Fecha": "max",  # Tomamos la última fecha dentro del mes
                }
            )
            .reset_index()
        )

        # Ordenar por fecha descendente
        df_agrupado = df_agrupado.sort_values(
            ["Centro", "Almacén", "Material", "Fecha"],
            ascending=[True, True, True, False],
        )

        # Para cada grupo (Centro/Almacén/Material), tomar los 2 últimos meses
        df_resultado = []

        for (centro, almacen, material), group in df_agrupado.groupby(
            ["Centro", "Almacén", "Material"]
        ):
            # Tomar los 2 últimos meses únicos
            meses_unicos = group.drop_duplicates("MesAno").head(2)

            if len(meses_unicos) >= 1:
                # Último mes
                ultimo_mes = meses_unicos.iloc[0]
                # Sumar todo lo facturado en ese mes
                facturacion_ultimo_mes = group[group["MesAno"] == ultimo_mes["MesAno"]]

                ultima_info = {
                    "Centro": centro,
                    "Almacén": almacen,
                    "Material": material,
                    "Ultima_Fecha_Facturacion": ultimo_mes["MesAno"],
                    "Ultima_Cantidad_Facturada": facturacion_ultimo_mes[
                        "Cantidad"
                    ].sum(),
                    "Ultimo_Importe_Facturado": facturacion_ultimo_mes["Importe"].sum(),
                }

                # Penúltimo mes (si existe)
                if len(meses_unicos) >= 2:
                    penultimo_mes = meses_unicos.iloc[1]
                    facturacion_penultimo_mes = group[
                        group["MesAno"] == penultimo_mes["MesAno"]
                    ]

                    ultima_info.update(
                        {
                            "Penultima_Fecha_Facturacion": penultimo_mes["MesAno"],
                            "Penultima_Cantidad_Facturada": facturacion_penultimo_mes[
                                "Cantidad"
                            ].sum(),
                            "Penultimo_Importe_Facturado": facturacion_penultimo_mes[
                                "Importe"
                            ].sum(),
                        }
                    )
                else:
                    # Si solo hay un mes, dejar penúltimas columnas vacías
                    ultima_info.update(
                        {
                            "Penultima_Fecha_Facturacion": "",
                            "Penultima_Cantidad_Facturada": 0,
                            "Penultimo_Importe_Facturado": 0,
                        }
                    )

                df_resultado.append(ultima_info)

        return pd.DataFrame(df_resultado)

    except Exception as e:
        logger.error(f"Error al calcular estadísticas de facturación: {str(e)}")
        return pd.DataFrame()


def procesar_datos_facturacion(df_facturacion: pd.DataFrame) -> pd.DataFrame:
    """
    Versión OPTIMIZADA del procesamiento de facturación.
    """
    if df_facturacion.empty:
        return pd.DataFrame()

    # Normalizar nombres de columnas - más eficiente
    df_facturacion.columns = [
        col.replace("Almacen", "Almacén").replace("Almaçen", "Almacén")
        for col in df_facturacion.columns
    ]

    # Diccionario de mapeo de patrones
    patrones = {
        "Solicitante": ["solicitante", "solicitud", "cliente solicitante"],
        "Razón Social": ["razón social", "razon social", "nombre cliente"],
        "Destinatario": ["destinatario", "cliente final", "destino"],
        "Fecha": ["fecha", "fecha factura", "fecha documento"],
        "Factura": ["factura", "no. factura", "documento"],
        "Material": ["material", "artículo", "producto"],
        "Texto Material": ["texto material", "descripción", "descripcion"],
        "Cantidad": ["cantidad", "qty", "quantity"],
        "UM": ["um", "unidad medida", "unidad"],
        "Importe": ["importe", "valor", "monto", "total"],
        "Centro": ["centro", "plant", "sede"],
        "Almacén": ["almacén", "almacen", "warehouse"],
        "Doc. Ventas": ["doc. ventas", "documento ventas", "pedido"],
        "Gpo. Vdor.": ["gpo. vdor.", "grupo vendedor", "vendedor"],
        "Grp. Cliente": ["grp. cliente", "grupo cliente", "tipo cliente"],
    }

    # Buscar columnas por patrones - optimizado
    mapeo_columnas = {}
    for col_requerida, patrones_list in patrones.items():
        if col_requerida not in df_facturacion.columns:
            for col in df_facturacion.columns:
                if any(patron in col.lower() for patron in patrones_list):
                    mapeo_columnas[col_requerida] = col
                    break
            if col_requerida not in mapeo_columnas:
                df_facturacion[col_requerida] = ""

    # Renombrar columnas según mapeo
    for col_dest, col_orig in mapeo_columnas.items():
        if col_orig in df_facturacion.columns:
            df_facturacion[col_dest] = df_facturacion[col_orig]

    # Normalizar IDs - vectorizado
    for col in ["Centro", "Material", "Almacén", "Destinatario", "Solicitante"]:
        if col in df_facturacion.columns:
            df_facturacion[col] = (
                df_facturacion[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0+$", "", regex=True)
            )

    # Convertir fechas - más eficiente
    if "Fecha" in df_facturacion.columns:
        df_facturacion["Fecha"] = pd.to_datetime(
            df_facturacion["Fecha"],
            dayfirst=True,
            errors="coerce",  # Agregar dayfirst=True
        )

    # Convertir numéricos - vectorizado
    for col in ["Cantidad", "Importe"]:
        if col in df_facturacion.columns:
            df_facturacion[col] = pd.to_numeric(
                df_facturacion[col], errors="coerce"
            ).fillna(0)

    return df_facturacion


def generar_reporte_consumo(df_facturacion: pd.DataFrame) -> pd.DataFrame:
    """
    Versión OPTIMIZADA del reporte de consumo con columna de consumo actual.
    Modificación: Asegura que el último mes y penúltimo mes sean diferentes.
    """
    if df_facturacion.empty:
        return pd.DataFrame()

    # Crear una barra de progreso para la generación del reporte
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Preparando datos de facturación...")

    # Eliminar duplicados y filtrar datos inválidos (más eficiente)
    df_facturacion = df_facturacion.drop_duplicates()

    # Filtrar solo registros con fecha válida; las cantidades pueden ser negativas (devoluciones)
    mask_fecha_valida = df_facturacion["Fecha"].notna()
    df_facturacion = df_facturacion[mask_fecha_valida].copy()

    if df_facturacion.empty:
        progress_bar.empty()
        status_text.empty()
        return pd.DataFrame()

    # Crear columnas auxiliares para cálculos rápidos (vectorizado)
    df_facturacion["AñoMes"] = df_facturacion["Fecha"].dt.to_period("M")

    # El precio unitario se calcula solo para registros con importe y cantidad positivos
    # Las cantidades negativas son devoluciones/créditos que afectan el consumo neto pero no el precio
    mask_precio_valido = (df_facturacion["Cantidad"] > 0) & (
        df_facturacion["Importe"] > 0
    )
    df_facturacion["PrecioUnitario"] = np.where(
        mask_precio_valido,
        df_facturacion["Importe"] / df_facturacion["Cantidad"],
        np.nan,
    )

    # Obtener el mes actual (mes corriente) para excluirlo de los cálculos
    mes_actual = pd.Timestamp.now().to_period("M")

    # Crear máscaras una sola vez
    mask_mes_actual = df_facturacion["AñoMes"] == mes_actual
    mask_historico = df_facturacion["AñoMes"] < mes_actual

    # Preparar datos para cálculos vectorizados
    status_text.text("Agrupando datos...")
    progress_bar.progress(0.1)

    # Obtener el último centro por destinatario (vectorizado) - UNA SOLA VEZ
    df_ultimo_centro = df_facturacion.sort_values(
        "Fecha", ascending=False
    ).drop_duplicates("Destinatario")[["Destinatario", "Centro", "Fecha"]]
    df_ultimo_centro["Ultima_compra_cliente"] = df_ultimo_centro["Fecha"].dt.strftime(
        "%m/%Y"
    )
    ultimo_centro_dict = df_ultimo_centro.set_index("Destinatario")["Centro"].to_dict()
    ultima_compra_dict = df_ultimo_centro.set_index("Destinatario")[
        "Ultima_compra_cliente"
    ].to_dict()

    # ============================================================
    # MODIFICACIÓN: Agregar columna de última facturación por Destinatario
    # ============================================================
    # Obtener última facturación por Destinatario (todas las combinaciones)
    df_ultima_fact_destinatario = df_facturacion.sort_values(
        "Fecha", ascending=False
    ).drop_duplicates("Destinatario")[["Destinatario", "Fecha"]]
    df_ultima_fact_destinatario["Ultima_facturacion_destinatario"] = (
        df_ultima_fact_destinatario["Fecha"].dt.strftime("%m/%Y")
    )
    ultima_fact_destinatario_dict = df_ultima_fact_destinatario.set_index(
        "Destinatario"
    )["Ultima_facturacion_destinatario"].to_dict()

    # Pre-calcular datos por grupo de manera vectorizada
    status_text.text("Calculando estadísticas por material...")
    progress_bar.progress(0.2)

    # Agrupar datos históricos UNA SOLA VEZ
    df_historico = df_facturacion[mask_historico]

    # Calcular consumo actual por grupo (vectorizado)
    df_mes_actual_grouped = (
        df_facturacion[mask_mes_actual]
        .groupby(["Solicitante", "Destinatario", "Material"])
        .agg(consumo_actual=("Cantidad", "sum"))
        .reset_index()
    )

    # Calcular estadísticas históricas por grupo
    df_historico_grouped = (
        df_historico.groupby(["Solicitante", "Destinatario", "Material"])
        .agg(
            cantidad_total_historico=("Cantidad", "sum"),  # neto (incluye devoluciones)
            fecha_min_historico=("Fecha", "min"),
            fecha_max_historico=("Fecha", "max"),
            meses_con_factura=("AñoMes", "nunique"),
            count_facturas=("Fecha", "count"),
        )
        .reset_index()
    )

    # Calcular precios por grupo (solo registros con precio válido = cantidad e importe > 0)
    df_para_precios = df_facturacion[
        (df_facturacion["Cantidad"] > 0) & (df_facturacion["Importe"] > 0)
    ]
    df_precios_grouped = (
        df_para_precios.groupby(["Solicitante", "Destinatario", "Material"])
        .agg(
            precio_min=(
                "PrecioUnitario",
                lambda x: x[x > 0].min() if (x > 0).any() else 0,
            ),
            precio_max=(
                "PrecioUnitario",
                lambda x: x[x > 0].max() if (x > 0).any() else 0,
            ),
            precio_prom=(
                "PrecioUnitario",
                lambda x: x[x > 0].mean() if (x > 0).any() else 0,
            ),
        )
        .reset_index()
    )

    # ============================================================
    # MODIFICACIÓN CRÍTICA: Obtener últimos dos MESES distintos (no facturas)
    # ============================================================
    status_text.text("Obteniendo últimos meses facturados...")
    progress_bar.progress(0.5)

    # Crear columnas de mes-año para agrupamiento
    df_facturacion["MesAno_str"] = df_facturacion["AñoMes"].dt.strftime("%m/%Y")
    df_facturacion["MesAno_num"] = (
        df_facturacion["AñoMes"].dt.year * 100 + df_facturacion["AñoMes"].dt.month
    )

    # Agrupar por grupo y mes para obtener totales mensuales
    monthly_totals = (
        df_facturacion.groupby(
            ["Solicitante", "Destinatario", "Material", "MesAno_num", "MesAno_str"]
        )
        .agg(
            Cantidad_mes=("Cantidad", "sum"),
            Importe_mes=("Importe", "sum"),
            Fecha_max=("Fecha", "max"),  # Tomar la fecha más reciente dentro del mes
        )
        .reset_index()
    )

    # Ordenar por grupo y mes (descendente) para obtener los últimos meses
    monthly_totals = monthly_totals.sort_values(
        ["Solicitante", "Destinatario", "Material", "MesAno_num"],
        ascending=[True, True, True, False],
    )

    # Para cada grupo, tomar los dos últimos meses DISTINTOS
    monthly_totals["orden"] = (
        monthly_totals.groupby(["Solicitante", "Destinatario", "Material"]).cumcount()
        + 1
    )

    # Filtrar solo los dos primeros meses distintos (último y penúltimo)
    df_ultimas_meses = monthly_totals[monthly_totals["orden"] <= 2].copy()

    # Pivotar para tener último y penúltimo mes en columnas separadas
    df_ultimas_pivot = df_ultimas_meses.pivot_table(
        index=["Solicitante", "Destinatario", "Material"],
        columns="orden",
        values=["MesAno_str", "Cantidad_mes", "Importe_mes", "Fecha_max"],
        aggfunc="first",
    )

    # Aplanar columnas
    df_ultimas_pivot.columns = [
        f"{col[0]}_{col[1]}" for col in df_ultimas_pivot.columns
    ]
    df_ultimas_pivot = df_ultimas_pivot.reset_index()

    # Calcular el precio unitario para el último y penúltimo mes
    df_ultimas_pivot["PrecioUnitario_1"] = np.where(
        df_ultimas_pivot["Cantidad_mes_1"] > 0,
        df_ultimas_pivot["Importe_mes_1"] / df_ultimas_pivot["Cantidad_mes_1"],
        0,
    )
    df_ultimas_pivot["PrecioUnitario_2"] = np.where(
        df_ultimas_pivot["Cantidad_mes_2"] > 0,
        df_ultimas_pivot["Importe_mes_2"] / df_ultimas_pivot["Cantidad_mes_2"],
        0,
    )

    # ============================================================
    # MODIFICACIÓN: Validar que las fechas sean diferentes
    # ============================================================
    # Si último mes y penúltimo mes son iguales, eliminar el penúltimo
    mask_mismos_meses = (
        df_ultimas_pivot["MesAno_str_1"] == df_ultimas_pivot["MesAno_str_2"]
    )
    df_ultimas_pivot.loc[
        mask_mismos_meses,
        [
            "MesAno_str_2",
            "Cantidad_mes_2",
            "Importe_mes_2",
            "PrecioUnitario_2",
            "Fecha_max_2",
        ],
    ] = ["", 0, 0, 0, pd.NaT]

    # Obtener datos básicos por grupo (primera fila)
    status_text.text("Preparando datos básicos...")
    progress_bar.progress(0.7)

    df_basicos = (
        df_facturacion.sort_values(
            ["Solicitante", "Destinatario", "Material", "Fecha"],
            ascending=[True, True, True, False],
        )
        .groupby(["Solicitante", "Destinatario", "Material"])
        .first()
        .reset_index()[
            [
                "Solicitante",
                "Destinatario",
                "Material",
                "Razón Social",
                "Texto Material",
                "UM",
                "Gpo. Vdor.",
                "Grp. Cliente",
            ]
        ]
    )

    # Combinar todos los datos
    status_text.text("Combinando datos...")
    progress_bar.progress(0.8)

    # Crear DataFrame base con todos los grupos únicos
    grupos_unicos = df_facturacion[
        ["Solicitante", "Destinatario", "Material"]
    ].drop_duplicates()

    # Combinar todos los datos usando merge
    reporte_final = grupos_unicos

    # Combinar con datos básicos
    reporte_final = pd.merge(
        reporte_final,
        df_basicos,
        on=["Solicitante", "Destinatario", "Material"],
        how="left",
    )

    # Combinar con datos históricos
    reporte_final = pd.merge(
        reporte_final,
        df_historico_grouped,
        on=["Solicitante", "Destinatario", "Material"],
        how="left",
    )

    # Combinar con consumo actual
    reporte_final = pd.merge(
        reporte_final,
        df_mes_actual_grouped,
        on=["Solicitante", "Destinatario", "Material"],
        how="left",
    )

    # Combinar con datos de precios
    reporte_final = pd.merge(
        reporte_final,
        df_precios_grouped,
        on=["Solicitante", "Destinatario", "Material"],
        how="left",
    )

    # Combinar con últimas facturas
    reporte_final = pd.merge(
        reporte_final,
        df_ultimas_pivot,
        on=["Solicitante", "Destinatario", "Material"],
        how="left",
    )

    # Agregar centro del último pedido
    reporte_final["Centro"] = reporte_final["Destinatario"].map(ultimo_centro_dict)
    reporte_final["Ultima_compra_cliente"] = reporte_final["Destinatario"].map(
        ultima_compra_dict
    )

    # ============================================================
    # MODIFICACIÓN: Agregar última facturación por Destinatario
    # ============================================================
    reporte_final["Ultima_facturacion_destinatario"] = reporte_final[
        "Destinatario"
    ].map(ultima_fact_destinatario_dict)

    # Calcular campos derivados (vectorizado)
    status_text.text("Calculando campos finales...")
    progress_bar.progress(0.9)

    # Calcular meses diferencia históricos
    reporte_final["meses_diff_historico"] = (
        reporte_final["fecha_max_historico"].dt.year
        - reporte_final["fecha_min_historico"].dt.year
    ) * 12 + (
        reporte_final["fecha_max_historico"].dt.month
        - reporte_final["fecha_min_historico"].dt.month
    )

    # Asegurar mínimo 1 mes
    reporte_final["meses_diff_historico"] = reporte_final["meses_diff_historico"].clip(
        lower=1
    )

    # Calcular consumo promedio mensual (solo histórico)
    reporte_final["Consumo_promedio_mensual"] = (
        (
            reporte_final["cantidad_total_historico"]
            / reporte_final["meses_diff_historico"]
        )
        .fillna(0)
        .astype(int)
    )

    # Calcular tendencia (solo histórico)
    reporte_final["Tendencia"] = (
        (reporte_final["meses_diff_historico"] / reporte_final["meses_con_factura"])
        .fillna(0)
        .round(2)
    )

    # Calcular tendencia de cantidad (solo histórico)
    reporte_final["Tendencia de cantidad"] = (
        (reporte_final["cantidad_total_historico"] / reporte_final["count_facturas"])
        .fillna(0)
        .round(2)
    )

    # Formatear fechas
    reporte_final["Ultimo mes facturacion"] = reporte_final["MesAno_str_1"]
    reporte_final["Penultima_fecha"] = reporte_final["MesAno_str_2"]

    # Renombrar columnas para el formato final
    reporte_final = reporte_final.rename(
        columns={
            "consumo_actual": "Consumo_actual",
            "Cantidad_mes_1": "Cantidad ultima",
            "Importe_mes_1": "Importe ultima",
            "PrecioUnitario_1": "Precio_unitario_ultima",
            "Cantidad_mes_2": "Cantidad_penultima",
            "Importe_mes_2": "Importe_penultima",
            "PrecioUnitario_2": "Precio_unitario_penultima",
        }
    )

    # Rellenar valores nulos
    for col in [
        "Consumo_actual",
        "Cantidad ultima",
        "Importe ultima",
        "Cantidad_penultima",
        "Importe_penultima",
        "cantidad_total_historico",
        "meses_con_factura",
        "count_facturas",
        "Precio_unitario_ultima",
        "Precio_unitario_penultima",
    ]:
        if col in reporte_final.columns:
            reporte_final[col] = reporte_final[col].fillna(0)

    for col in [
        "Ultimo mes facturacion",
        "Penultima_fecha",
        "Ultima_compra_cliente",
        "Ultima_facturacion_destinatario",  # Nueva columna
        "Razón Social",
        "Texto Material",
        "UM",
        "Gpo. Vdor.",
        "Grp. Cliente",
        "Centro",
    ]:
        if col in reporte_final.columns:
            reporte_final[col] = reporte_final[col].fillna("")

    # Ordenar columnas según lo solicitado (agregando la nueva columna)
    columnas_orden = [
        "Centro",
        "Grp. Cliente",
        "Gpo. Vdor.",
        "Solicitante",
        "Destinatario",
        "Razón Social",
        "Material",
        "Texto Material",
        "Ultima_compra_cliente",
        "Ultima_facturacion_destinatario",  # Nueva columna
        "Consumo_promedio_mensual",
        "Consumo_actual",
        "UM",
        "Tendencia",
        "Tendencia de cantidad",
        "Ultimo mes facturacion",
        "Cantidad ultima",
        "Importe ultima",
        "Precio_unitario_ultima",
        "Penultima_fecha",
        "Cantidad_penultima",
        "Importe_penultima",
        "Precio_unitario_penultima",
        "precio_min",
        "precio_max",
        "precio_prom",
    ]

    # Crear columnas que puedan faltar
    for col in columnas_orden:
        if col not in reporte_final.columns:
            if col in [
                "Centro",
                "Grp. Cliente",
                "Gpo. Vdor.",
                "Solicitante",
                "Destinatario",
                "Razón Social",
                "Material",
                "Texto Material",
                "UM",
                "Ultima_facturacion_destinatario",
                "Ultima_compra_cliente",
            ]:
                reporte_final[col] = ""
            elif col == "Consumo_actual":
                reporte_final[col] = 0
            else:
                reporte_final[col] = 0

    # Limpiar barra de progreso
    progress_bar.progress(1.0)
    progress_bar.empty()
    status_text.empty()

    return reporte_final[columnas_orden]


# =========================
# MODIFICAR: Función obtener_disponible_por_fuente para manejar lotes específicos
# =========================
def obtener_disponible_por_fuente(
    fuente: str,
    material: str,
    centro: str,
    almacen: str,
    df_fuente: pd.DataFrame,
    inventario_df: pd.DataFrame,
    lote: str = "",
) -> float:
    """Obtiene la cantidad disponible según el tipo de fuente y lote específico."""

    if fuente == "Corta caducidad":
        # Para Corta caducidad: usar "Libre Utilización" del inventario por lote específico
        if inventario_df is None or inventario_df.empty:
            return 0.0

        # Buscar en el inventario para el material, centro, almacén y lote específicos
        # NOTA: El inventario general no tiene información de lote, así que usamos el valor de la hoja externa
        # que ya tiene la cantidad por lote en "CantidadDisp"
        if df_fuente is None or df_fuente.empty:
            return 0.0

        # Filtrar por material, centro, almacén y lote específico
        filtro = (
            (df_fuente["Material"] == material)
            & (df_fuente["Centro"] == centro)
            & (df_fuente["Almacén"] == almacen)
            & (df_fuente["Lote"] == lote)
        )

        disponible = df_fuente[filtro]["CantidadDisp"].sum()
        return float(disponible)

    elif fuente in ["Cosmopark", "PNC", "Caduco"]:
        # Para Cosmopark y PNC: usar "CantidadDisp" de la hoja externa por lote específico
        if df_fuente is None or df_fuente.empty:
            return 0.0

        # Filtrar por material, centro, almacén y lote específico
        filtro = (
            (df_fuente["Material"] == material)
            & (df_fuente["Centro"] == centro)
            & (df_fuente["Almacén"] == almacen)
        )

        # Si hay lote, agregar filtro por lote
        if lote:
            filtro = filtro & (df_fuente["Lote"] == lote)

        disponible = df_fuente[filtro]["CantidadDisp"].sum()
        return float(disponible)

    elif fuente in ["Lento mov", "Sustituto"]:
        # Para Lento mov y Sustituto: usar inventario filtrado 1030/1031 (sin lote específico)
        inventario_filtrado = get_inventory_by_all_centers_filtered_1030_1031(
            inventario_df, material
        )
        return sum(inventario_filtrado.values())

    else:
        return 0.0


# =========================
# Función para obtener tránsito por almacén
# =========================
def get_transito_by_centro_almacen(
    inventario_df: pd.DataFrame, centro: str, material: str
) -> Dict[str, float]:
    """Obtiene la cantidad en tránsito de un material por almacén para un centro específico."""
    if inventario_df is None or inventario_df.empty:
        return {"1030": 0.0, "1031": 0.0, "1032": 0.0}

    try:
        # Filtrar por centro y material
        df_material = inventario_df[
            (inventario_df["Centro"] == centro)
            & (inventario_df["Material"] == material)
        ]
        if df_material.empty:
            return {"1030": 0.0, "1031": 0.0, "1032": 0.0}

        transito_por_almacen = {}
        for almacen in ["1030", "1031", "1032"]:
            transito = df_material[df_material["Almacén"] == almacen][
                "Cant. en Tránsito"
            ].sum()
            transito_por_almacen[almacen] = float(transito)

        return transito_por_almacen
    except Exception as e:
        logger.error(f"Error en get_transito_by_centro_almacen: {str(e)}")
        return {"1030": 0.0, "1031": 0.0, "1032": 0.0}


# =========================
# MODIFICAR: Función para obtener tránsito total por centro
# =========================
def get_transito_total_centro(
    inventario_df: pd.DataFrame, centro: str, material: str
) -> float:
    """Obtiene la cantidad total en tránsito de un material para un centro específico (suma de los 3 almacenes)."""
    transito_por_almacen = get_transito_by_centro_almacen(
        inventario_df, centro, material
    )
    return sum(transito_por_almacen.values())


def obtener_inventario_por_centro(
    inventario_df: pd.DataFrame, material: str
) -> Dict[str, float]:
    """Obtiene inventario de un material por centro (solo para centros específicos)"""
    if inventario_df is None or inventario_df.empty:
        return {}

    df_material = inventario_df[inventario_df["Material"] == material]
    if df_material.empty:
        return {}

    inventario_por_centro = {}
    # Solo para centros específicos (no para 1030, 1031, 1032 que manejamos por separado)
    for centro in ["1001", "1003", "1004", "1017", "1018", "1022", "1036"]:
        disponible = df_material[df_material["Centro"] == centro][
            "Libre Utilización"
        ].sum()
        inventario_por_centro[centro] = float(disponible)

    return inventario_por_centro


def obtener_inventario_por_centro_y_almacen(
    inventario_df: pd.DataFrame, centro: str, material: str
) -> Dict[str, Dict[str, float]]:
    """Obtiene inventario de un material por centro y almacén específicos"""
    if inventario_df is None or inventario_df.empty:
        return {}

    # Filtrar por centro y material
    df_filtrado = inventario_df[
        (inventario_df["Centro"] == centro) & (inventario_df["Material"] == material)
    ]

    if df_filtrado.empty:
        return {}

    # Agrupar por almacén para obtener el inventario disponible
    inventario_por_almacen = {}
    for almacen in ["1030", "1031", "1032", "1060"]:
        disponible = df_filtrado[df_filtrado["Almacén"] == almacen][
            "Libre Utilización"
        ].sum()
        inventario_por_almacen[almacen] = float(disponible)

    return {centro: inventario_por_almacen}


# =========================
# Función para obtener inventario total de todos los centros
# =========================
def get_inventory_by_all_centers(
    inventario_df: pd.DataFrame, material: str
) -> Dict[str, float]:
    """Obtiene el inventario de un material en todos los centros disponibles."""
    if inventario_df is None or inventario_df.empty:
        return {}

    try:
        df_material = inventario_df[inventario_df["Material"] == material]
        if df_material.empty:
            return {}

        inventory_by_center = (
            df_material.groupby("Centro")["Libre Utilización"].sum().to_dict()
        )
        return {str(center): float(qty) for center, qty in inventory_by_center.items()}
    except Exception as e:
        logger.error(f"Error en get_inventory_by_all_centers: {str(e)}")
        return {}


# =========================
# Nueva función: Obtener inventario por centro solo para almacenes 1030 y 1031
# =========================
def get_inventory_by_all_centers_filtered_1030_1031(
    inventario_df: pd.DataFrame, material: str
) -> Dict[str, float]:
    """Obtiene el inventario de un material en todos los centros, sumando solo los almacenes 1030 y 1031."""
    if inventario_df is None or inventario_df.empty:
        return {}

    try:
        # Filtrar por material y almacenes 1030 o 1031
        df_material = inventario_df[
            (inventario_df["Material"] == material)
            & (inventario_df["Almacén"].isin(["1030", "1031", "1060"]))
        ]

        if df_material.empty:
            return {}

        # Agrupar por centro y sumar el inventario disponible
        inventory_by_center = (
            df_material.groupby("Centro")["Libre Utilización"].sum().to_dict()
        )
        return {str(center): float(qty) for center, qty in inventory_by_center.items()}
    except Exception as e:
        logger.error(
            f"Error en get_inventory_by_all_centers_filtered_1030_1031: {str(e)}"
        )
        return {}


# =========================
# MODIFICAR: función crear_linea_sugerencia para usar tránsito por centro
# =========================
def crear_linea_sugerencia(
    pedido: pd.Series,
    material_sugerido: str,
    fuente: str,
    centro_sugerido: str,
    almacen_sugerido: str,
    disponible: float,
    inventario_df: pd.DataFrame,
    lote: str = "",
    fecha_caducidad: str = "",
    descripcion_sugerida: str = "",
) -> Dict:
    """Crea una línea de sugerencia con el formato requerido"""

    # Obtener el centro del pedido
    centro_pedido = str(pedido.get("Centro", "")).strip()
    material_solicitado = str(pedido.get("Material", "")).strip()

    # Determinar qué material usar para los cálculos de inventario
    if "Sustituto" in fuente:
        material_para_inventario = material_sugerido
    else:
        material_para_inventario = material_solicitado

    # Obtener inventario del material (sustituto o solicitado) en el centro del pedido por almacén
    inventario_centro_almacen = obtener_inventario_por_centro_y_almacen(
        inventario_df, centro_pedido, material_para_inventario
    )

    # MODIFICACIÓN: Para las columnas Inv 1001, Inv 1003, etc. usar material_sugerido si está disponible,
    # de lo contrario usar material_solicitado. Sumar solo almacenes 1030/1031 por centro.
    material_para_columnas_inv = (
        material_sugerido if material_sugerido else material_solicitado
    )

    # Crear diccionario para almacenar inventario por centro (solo almacenes 1030/1031)
    inventario_por_centro_filtrado = {}

    if inventario_df is not None and not inventario_df.empty:
        # Filtrar por material específico
        df_material = inventario_df[
            inventario_df["Material"] == material_para_columnas_inv
        ]

        if not df_material.empty:
            # Filtrar solo almacenes 1030, 1031 y 1060
            df_material = df_material[
                df_material["Almacén"].isin(["1030", "1031", "1060"])
            ]

            # Agrupar por centro y sumar el inventario disponible
            inventario_por_centro_filtrado = (
                df_material.groupby("Centro")["Libre Utilización"].sum().to_dict()
            )

    # Obtener tránsito por almacén para el centro específico y material correcto
    transito_por_almacen = get_transito_by_centro_almacen(
        inventario_df, centro_pedido, material_para_inventario
    )

    # Obtener tránsito total para el centro específico
    transito_total = get_transito_total_centro(
        inventario_df, centro_pedido, material_para_inventario
    )

    # Calcular cantidad a ofertar (mínimo entre pendiente y disponible)
    cantidad_pendiente = float(pedido.get("Pendiente", 0))
    cantidad_ofertar = (
        min(cantidad_pendiente, disponible) if cantidad_pendiente > 0 else 0
    )

    # Calcular bloqueado
    bloqueado_val = ""
    if "Sts. Créd." in pedido and str(pedido["Sts. Créd."]).strip() == "B":
        bloqueado_val = "Crédito"
    if "Bloqueo Ent." in pedido and str(pedido["Bloqueo Ent."]).strip() not in [
        "",
        "nan",
    ]:
        if bloqueado_val:
            bloqueado_val = "Detenido por ambos"
        else:
            bloqueado_val = "Detenido"

    # Formatear fecha de caducidad
    if fecha_caducidad:
        try:
            if isinstance(fecha_caducidad, str) and fecha_caducidad.strip():
                fecha_dt = pd.to_datetime(
                    fecha_caducidad, dayfirst=True, errors="coerce"
                )
                if pd.notnull(fecha_dt):
                    fecha_caducidad = fecha_dt.strftime("%d/%m/%Y")
                else:
                    fecha_caducidad = ""
            elif isinstance(fecha_caducidad, (pd.Timestamp, datetime.datetime)):
                fecha_caducidad = fecha_caducidad.strftime("%d/%m/%Y")
            else:
                fecha_caducidad = ""
        except Exception:
            fecha_caducidad = ""

    # Obtener inventario específico por almacén para el centro del pedido
    inv_1030 = 0
    inv_1031 = 0
    inv_1032 = 0
    inv_1060 = 0

    if centro_pedido in inventario_centro_almacen:
        almacenes = inventario_centro_almacen[centro_pedido]
        inv_1030 = almacenes.get("1030", 0)
        inv_1031 = almacenes.get("1031", 0)
        inv_1032 = almacenes.get("1032", 0)
        inv_1060 = almacenes.get("1060", 0)

    # Calcular disponibilidad en centro 1031 para almacenes 1030 y 1032
    disp_1031_1030 = 0
    disp_1031_1032 = 0
    if inventario_df is not None and not inventario_df.empty:
        # Disponible en centro 1031, almacén 1030
        disp_1031_1030 = inventario_df[
            (inventario_df["Centro"] == "1031")
            & (inventario_df["Almacén"] == "1030")
            & (inventario_df["Material"] == material_para_inventario)
        ]["Libre Utilización"].sum()

        # Disponible en centro 1031, almacén 1032
        disp_1031_1032 = inventario_df[
            (inventario_df["Centro"] == "1031")
            & (inventario_df["Almacén"] == "1032")
            & (inventario_df["Material"] == material_para_inventario)
        ]["Libre Utilización"].sum()

    # Construir la línea
    linea = {
        Columnas.GRUPO_CLIENTE: str(pedido.get("Gpo. Cte.", "")).strip(),
        Columnas.FECHA: pedido.get("Fecha", ""),
        Columnas.PEDIDO: pedido.get("Pedido", ""),
        Columnas.GRUPO_VENDEDOR: pedido.get("Gpo.Vdor.", ""),
        Columnas.SOLICITANTE: pedido.get("Solicitante", ""),
        Columnas.DESTINATARIO: pedido.get("Destinatario", ""),
        Columnas.RAZON_SOCIAL: str(pedido.get("Razón Social", "")),
        Columnas.CENTRO_PEDIDO: centro_pedido,
        Columnas.ALMACEN: str(pedido.get("Almacén", "")).strip(),
        Columnas.MATERIAL_SOLICITADO: material_solicitado,
        Columnas.MATERIAL_BASE: material_solicitado,
        Columnas.DESCRIPCION_SOLICITADA: str(pedido.get("Texto Material", "")),
        Columnas.CANTIDAD_PEDIDO: pedido.get("Cantidad", ""),
        Columnas.CANTIDAD_PENDIENTE: cantidad_pendiente,
        Columnas.CANTIDAD_OFERTAR: cantidad_ofertar,
        Columnas.PRECIO: pedido.get("Precio", 0),
        Columnas.FUENTE: fuente,
        Columnas.MATERIAL_SUGERIDO: material_sugerido,
        Columnas.DESCRIPCION_SUGERIDA: descripcion_sugerida,
        Columnas.CENTRO_SUGERIDO: centro_sugerido,
        Columnas.ALMACEN_SUGERIDO: almacen_sugerido,
        Columnas.DISPONIBLE: disponible,
        Columnas.LOTE: lote,
        Columnas.FECHA_CADUCIDAD: fecha_caducidad,
        Columnas.CENTRO_INV: centro_pedido,
        Columnas.INV_1030: inv_1030,
        Columnas.INV_1031: inv_1031,
        Columnas.INV_1032: inv_1032,
        Columnas.INV_1060: inv_1060,
        Columnas.MESES_INVENTARIO: 0.0,  # se calcula en post-proceso
        Columnas.PROMEDIO_CONSUMO_12M: 0.0,  # se calcula en post-proceso
        Columnas.CONSUMO_DESTINATARIO_12M: 0.0,  # se calcula en post-proceso
        Columnas.CANT_TRANSITO: transito_total,
        Columnas.CANT_TRANSITO_1030: transito_por_almacen.get("1030", 0),
        Columnas.CANT_TRANSITO_1031: transito_por_almacen.get("1031", 0),
        Columnas.CANT_TRANSITO_1032: transito_por_almacen.get("1032", 0),
        Columnas.DISP_1031_1030: disp_1031_1030,
        Columnas.DISP_1031_1032: disp_1031_1032,
        # MODIFICACIÓN: Usar inventario_por_centro_filtrado que suma solo almacenes 1030/1031
        Columnas.INV_1001: inventario_por_centro_filtrado.get("1001", 0),
        Columnas.INV_1003: inventario_por_centro_filtrado.get("1003", 0),
        Columnas.INV_1004: inventario_por_centro_filtrado.get("1004", 0),
        Columnas.INV_1017: inventario_por_centro_filtrado.get("1017", 0),
        Columnas.INV_1018: inventario_por_centro_filtrado.get("1018", 0),
        Columnas.INV_1022: inventario_por_centro_filtrado.get("1022", 0),
        Columnas.INV_1036: inventario_por_centro_filtrado.get("1036", 0),
        Columnas.BLOQUEADO: bloqueado_val,
    }

    return linea


# =========================
# MODIFICAR: crear_linea_sin_sugerencia para usar tránsito por centro
# =========================
def crear_linea_sin_sugerencia(pedido: pd.Series, inventario_df: pd.DataFrame) -> Dict:
    """Crea una línea sin sugerencia (fuente vacía) para mostrar datos originales"""

    # Obtener el centro del pedido
    centro_pedido = str(pedido.get("Centro", "")).strip()
    material_solicitado = str(pedido.get("Material", "")).strip()

    # Para líneas sin sugerencia, usar el material solicitado para las columnas de inventario
    material_para_inventario = material_solicitado

    # Obtener inventario del material SOLICITADO en el centro del pedido por almacén
    inventario_centro_almacen = obtener_inventario_por_centro_y_almacen(
        inventario_df, centro_pedido, material_para_inventario
    )

    # MODIFICACIÓN: Para líneas sin sugerencia, usar material_solicitado para columnas Inv 1001, etc.
    # Sumar solo almacenes 1030/1031 por centro
    inventario_por_centro_filtrado = {}

    if inventario_df is not None and not inventario_df.empty:
        # Filtrar por material solicitado
        df_material = inventario_df[inventario_df["Material"] == material_solicitado]

        if not df_material.empty:
            # Filtrar solo almacenes 1030, 1031 y 1060
            df_material = df_material[
                df_material["Almacén"].isin(["1030", "1031", "1060"])
            ]

            # Agrupar por centro y sumar el inventario disponible
            inventario_por_centro_filtrado = (
                df_material.groupby("Centro")["Libre Utilización"].sum().to_dict()
            )

    # Obtener tránsito por almacén para el centro específico
    transito_por_almacen = get_transito_by_centro_almacen(
        inventario_df, centro_pedido, material_para_inventario
    )

    # Obtener tránsito total para el centro específico
    transito_total = get_transito_total_centro(
        inventario_df, centro_pedido, material_para_inventario
    )

    # Calcular bloqueado
    bloqueado_val = ""
    if "Sts. Créd." in pedido and str(pedido["Sts. Créd."]).strip() == "B":
        bloqueado_val = "Crédito"
    if "Bloqueo Ent." in pedido and str(pedido["Bloqueo Ent."]).strip() not in [
        "",
        "nan",
    ]:
        if bloqueado_val:
            bloqueado_val = "Detenido por ambos"
        else:
            bloqueado_val = "Detenido"

    # Obtener inventario específico por almacén para el centro del pedido
    inv_1030 = 0
    inv_1031 = 0
    inv_1032 = 0
    inv_1060 = 0

    if centro_pedido in inventario_centro_almacen:
        almacenes = inventario_centro_almacen[centro_pedido]
        inv_1030 = almacenes.get("1030", 0)
        inv_1031 = almacenes.get("1031", 0)
        inv_1032 = almacenes.get("1032", 0)
        inv_1060 = almacenes.get("1060", 0)

    # Calcular disponibilidad en centro 1031 para almacenes 1030 y 1032
    disp_1031_1030 = 0
    disp_1031_1032 = 0
    if inventario_df is not None and not inventario_df.empty:
        # Disponible en centro 1031, almacén 1030
        disp_1031_1030 = inventario_df[
            (inventario_df["Centro"] == "1031")
            & (inventario_df["Almacén"] == "1030")
            & (inventario_df["Material"] == material_para_inventario)
        ]["Libre Utilización"].sum()

        # Disponible en centro 1031, almacén 1032
        disp_1031_1032 = inventario_df[
            (inventario_df["Centro"] == "1031")
            & (inventario_df["Almacén"] == "1032")
            & (inventario_df["Material"] == material_para_inventario)
        ]["Libre Utilización"].sum()

    # Construir la línea
    linea = {
        Columnas.GRUPO_CLIENTE: str(pedido.get("Gpo. Cte.", "")).strip(),
        Columnas.FECHA: pedido.get("Fecha", ""),
        Columnas.PEDIDO: pedido.get("Pedido", ""),
        Columnas.GRUPO_VENDEDOR: pedido.get("Gpo.Vdor.", ""),
        Columnas.SOLICITANTE: pedido.get("Solicitante", ""),
        Columnas.DESTINATARIO: pedido.get("Destinatario", ""),
        Columnas.RAZON_SOCIAL: str(pedido.get("Razón Social", "")),
        Columnas.CENTRO_PEDIDO: centro_pedido,
        Columnas.ALMACEN: str(pedido.get("Almacén", "")).strip(),
        Columnas.MATERIAL_SOLICITADO: material_solicitado,
        Columnas.MATERIAL_BASE: material_solicitado,
        Columnas.DESCRIPCION_SOLICITADA: str(pedido.get("Texto Material", "")),
        Columnas.CANTIDAD_PEDIDO: pedido.get("Cantidad", ""),
        Columnas.CANTIDAD_PENDIENTE: float(pedido.get("Pendiente", 0)),
        Columnas.CANTIDAD_OFERTAR: 0,
        Columnas.PRECIO: pedido.get("Precio", 0),
        Columnas.FUENTE: "",
        Columnas.MATERIAL_SUGERIDO: "",
        Columnas.DESCRIPCION_SUGERIDA: "",
        Columnas.CENTRO_SUGERIDO: "",
        Columnas.ALMACEN_SUGERIDO: "",
        Columnas.DISPONIBLE: 0,
        Columnas.LOTE: "",
        Columnas.FECHA_CADUCIDAD: "",
        Columnas.CENTRO_INV: centro_pedido,
        Columnas.INV_1030: inv_1030,
        Columnas.INV_1031: inv_1031,
        Columnas.INV_1032: inv_1032,
        Columnas.INV_1060: inv_1060,
        Columnas.MESES_INVENTARIO: 0.0,  # se calcula en post-proceso
        Columnas.PROMEDIO_CONSUMO_12M: 0.0,  # se calcula en post-proceso
        Columnas.CONSUMO_DESTINATARIO_12M: 0.0,  # se calcula en post-proceso
        Columnas.CANT_TRANSITO: transito_total,
        Columnas.CANT_TRANSITO_1030: transito_por_almacen.get("1030", 0),
        Columnas.CANT_TRANSITO_1031: transito_por_almacen.get("1031", 0),
        Columnas.CANT_TRANSITO_1032: transito_por_almacen.get("1032", 0),
        Columnas.DISP_1031_1030: disp_1031_1030,
        Columnas.DISP_1031_1032: disp_1031_1032,
        # MODIFICACIÓN: Usar inventario_por_centro_filtrado que suma solo almacenes 1030/1031
        Columnas.INV_1001: inventario_por_centro_filtrado.get("1001", 0),
        Columnas.INV_1003: inventario_por_centro_filtrado.get("1003", 0),
        Columnas.INV_1004: inventario_por_centro_filtrado.get("1004", 0),
        Columnas.INV_1017: inventario_por_centro_filtrado.get("1017", 0),
        Columnas.INV_1018: inventario_por_centro_filtrado.get("1018", 0),
        Columnas.INV_1022: inventario_por_centro_filtrado.get("1022", 0),
        Columnas.INV_1036: inventario_por_centro_filtrado.get("1036", 0),
        Columnas.BLOQUEADO: bloqueado_val,
    }

    return linea


# =========================
# NUEVAS FUNCIONES: consolidar sugerencias repetidas y agrupar fuentes
# =========================
def unir_fuentes_repetidas(fuentes: pd.Series) -> str:
    """Une fuentes repetidas preservando el orden y evitando duplicados internos."""
    resultado = []
    vistos = set()

    for fuente in fuentes.fillna(""):
        for parte in [p.strip() for p in str(fuente).split("/") if p.strip()]:
            clave = parte.casefold()
            if clave not in vistos:
                vistos.add(clave)
                resultado.append(parte)

    return "/".join(resultado)


def consolidar_sugerencias_repetidas(df_resultado: pd.DataFrame) -> pd.DataFrame:
    """Consolida sugerencias idénticas y solo agrupa la columna Fuente."""
    if (
        df_resultado is None
        or df_resultado.empty
        or Columnas.FUENTE not in df_resultado.columns
    ):
        return df_resultado

    df = df_resultado.copy()
    df["_orden_original"] = np.arange(len(df))

    mask_sugerencias = df[Columnas.FUENTE].fillna("").astype(str).str.strip() != ""
    if not mask_sugerencias.any():
        return df.drop(columns=["_orden_original"])

    df_sin_sugerencia = df[~mask_sugerencias].copy()
    df_con_sugerencia = df[mask_sugerencias].copy()

    columnas_clave = [
        col
        for col in df_con_sugerencia.columns
        if col not in [Columnas.FUENTE, "_orden_original"]
    ]

    df_consolidado = (
        df_con_sugerencia.groupby(columnas_clave, dropna=False, as_index=False)
        .agg(
            {
                Columnas.FUENTE: unir_fuentes_repetidas,
                "_orden_original": "min",
            }
        )
        .sort_values("_orden_original")
    )

    df_final = pd.concat(
        [df_sin_sugerencia, df_consolidado],
        ignore_index=True,
        sort=False,
    ).sort_values("_orden_original")

    return df_final.drop(columns=["_orden_original"]).reset_index(drop=True)


# =========================
# MODIFICAR: función buscar_sugerencias_exactas para manejar lotes específicos
# =========================
def buscar_sugerencias_exactas(
    pedido: pd.Series,
    hojas_externas: Dict[str, pd.DataFrame],
    fuentes_activas: List[str],
    inventario_df: pd.DataFrame,
) -> List[Dict]:
    """Busca sugerencias exactas (1:1) en las hojas externas según nuevas reglas."""
    sugerencias = []
    material_solicitado = str(pedido.get("Material", "")).strip()

    if not material_solicitado:
        return sugerencias

    # Para cada fuente activa
    for fuente in fuentes_activas:
        if fuente not in hojas_externas:
            continue

        df_fuente = hojas_externas[fuente]

        # VERIFICACIÓN DE SEGURIDAD: Asegurar que la columna Material existe
        if "Material" not in df_fuente.columns:
            logger.warning(
                f"La hoja '{fuente}' no tiene columna 'Material'. Se omitirá."
            )
            continue

        if df_fuente.empty:
            continue

        if fuente == "Sustituto":
            # Buscar sustitutos para el material solicitado
            sustitutos = df_fuente[df_fuente["Material"] == material_solicitado]

            for _, sustituto_row in sustitutos.iterrows():
                material_sustituto = str(
                    sustituto_row.get("Material sustituto", "")
                ).strip()
                if not material_sustituto:
                    continue

                # Buscar el material sustituto en otras fuentes
                otras_fuentes = [
                    f for f in fuentes_activas if f not in ["Sustituto", "Lento mov"]
                ]
                encontrado_en_otras = False

                for otra_fuente in otras_fuentes:
                    if otra_fuente in hojas_externas:
                        df_otra = hojas_externas[otra_fuente]
                        coincidencias = df_otra[
                            df_otra["Material"] == material_sustituto
                        ]

                        if not coincidencias.empty:
                            encontrado_en_otras = True
                            # Crear una línea por cada coincidencia en esta otra fuente
                            for _, coincidencia in coincidencias.iterrows():
                                # Obtener detalles de la coincidencia
                                centro = str(coincidencia.get("Centro", "")).strip()
                                almacen = str(coincidencia.get("Almacén", "")).strip()
                                lote = str(coincidencia.get("Lote", "")).strip()
                                fecha_cad = coincidencia.get("FechaCaducidad", "")

                                # Calcular disponible según el tipo de fuente combinada
                                disponible_fuente = obtener_disponible_por_fuente(
                                    fuente=otra_fuente,
                                    material=material_sustituto,
                                    centro=centro,
                                    almacen=almacen,
                                    df_fuente=df_otra,
                                    inventario_df=inventario_df,
                                    lote=lote,  # Pasamos el lote específico
                                )

                                # Omitir sugerencias sin disponible
                                if disponible_fuente <= 0:
                                    continue

                                # Formatear fecha si es necesario
                                fecha_cad = formatear_fecha_caducidad(fecha_cad)

                                # Crear línea con fuente combinada
                                fuente_combinada = f"Sustituto/{otra_fuente}"
                                linea = crear_linea_sugerencia(
                                    pedido=pedido,
                                    material_sugerido=material_sustituto,
                                    fuente=fuente_combinada,
                                    centro_sugerido=centro,
                                    almacen_sugerido=almacen,
                                    disponible=disponible_fuente,
                                    inventario_df=inventario_df,
                                    lote=lote,
                                    fecha_caducidad=fecha_cad,
                                    descripcion_sugerida=str(
                                        sustituto_row.get(
                                            "Texto material sustituto", ""
                                        )
                                    ),
                                )
                                sugerencias.append(linea)

                # Si no se encontró en ninguna otra fuente, crear una línea solo con Sustituto
                if not encontrado_en_otras:
                    # Para Sustituto solo, usar inventario filtrado por 1030/1031
                    inventario_filtrado = (
                        get_inventory_by_all_centers_filtered_1030_1031(
                            inventario_df, material_sustituto
                        )
                    )
                    disponible_fuente = sum(inventario_filtrado.values())

                    # Omitir sugerencias sin disponible
                    if disponible_fuente > 0:
                        linea = crear_linea_sugerencia(
                            pedido=pedido,
                            material_sugerido=material_sustituto,
                            fuente="Sustituto",
                            centro_sugerido="",
                            almacen_sugerido="",
                            disponible=disponible_fuente,
                            inventario_df=inventario_df,
                            descripcion_sugerida=str(
                                sustituto_row.get("Texto material sustituto", "")
                            ),
                        )
                        sugerencias.append(linea)

        elif fuente == "Lento mov":
            # Buscar el material solicitado en Lento mov
            coincidencias = df_fuente[df_fuente["Material"] == material_solicitado]

            if not coincidencias.empty:
                # Buscar en otras fuentes (excluyendo Sustituto y Lento mov)
                otras_fuentes = [
                    f for f in fuentes_activas if f not in ["Sustituto", "Lento mov"]
                ]
                encontrado_en_otras = False

                for otra_fuente in otras_fuentes:
                    if otra_fuente in hojas_externas:
                        df_otra = hojas_externas[otra_fuente]
                        coincidencias_otra = df_otra[
                            df_otra["Material"] == material_solicitado
                        ]

                        if not coincidencias_otra.empty:
                            encontrado_en_otras = True
                            # Crear una línea por cada coincidencia en esta otra fuente
                            for _, coincidencia_otra in coincidencias_otra.iterrows():
                                # Combinar fuentes
                                fuente_combinada = f"Lento mov/{otra_fuente}"

                                # Obtener detalles de la coincidencia
                                centro = str(
                                    coincidencia_otra.get("Centro", "")
                                ).strip()
                                almacen = str(
                                    coincidencia_otra.get("Almacén", "")
                                ).strip()
                                lote = str(coincidencia_otra.get("Lote", "")).strip()
                                fecha_cad = coincidencia_otra.get("FechaCaducidad", "")

                                # Calcular disponible según el tipo de fuente combinada
                                disponible_fuente = obtener_disponible_por_fuente(
                                    fuente=otra_fuente,
                                    material=material_solicitado,
                                    centro=centro,
                                    almacen=almacen,
                                    df_fuente=df_otra,
                                    inventario_df=inventario_df,
                                    lote=lote,  # Pasamos el lote específico
                                )

                                # Omitir sugerencias sin disponible
                                if disponible_fuente <= 0:
                                    continue

                                # Formatear fecha si es necesario
                                fecha_cad = formatear_fecha_caducidad(fecha_cad)

                                linea = crear_linea_sugerencia(
                                    pedido=pedido,
                                    material_sugerido=material_solicitado,
                                    fuente=fuente_combinada,
                                    centro_sugerido=centro,
                                    almacen_sugerido=almacen,
                                    disponible=disponible_fuente,
                                    inventario_df=inventario_df,
                                    lote=lote,
                                    fecha_caducidad=fecha_cad,
                                )
                                sugerencias.append(linea)
                            break  # Solo una combinación por tipo de fuente

                if not encontrado_en_otras:
                    # Para Lento mov solo, usar inventario filtrado por 1030/1031
                    inventario_filtrado = (
                        get_inventory_by_all_centers_filtered_1030_1031(
                            inventario_df, material_solicitado
                        )
                    )
                    disponible_fuente = sum(inventario_filtrado.values())

                    # Omitir sugerencias sin disponible
                    if disponible_fuente > 0:
                        linea = crear_linea_sugerencia(
                            pedido=pedido,
                            material_sugerido=material_solicitado,
                            fuente="Lento mov",
                            centro_sugerido="",
                            almacen_sugerido="",
                            disponible=disponible_fuente,
                            inventario_df=inventario_df,
                        )
                        sugerencias.append(linea)

        else:
            # Para otras fuentes (Corta caducidad, Cosmopark, PNC, Caduco)
            coincidencias = df_fuente[df_fuente["Material"] == material_solicitado]

            for _, coincidencia in coincidencias.iterrows():
                centro = str(coincidencia.get("Centro", "")).strip()
                almacen = str(coincidencia.get("Almacén", "")).strip()
                lote = str(coincidencia.get("Lote", "")).strip()
                fecha_cad = coincidencia.get("FechaCaducidad", "")

                # Para PNC: usar directamente CantidadDisp del registro (columna "Cantidad")
                # evitando re-filtrados que pueden dar 0 por datos incompletos de Centro/Almacén
                if fuente == "PNC":
                    disponible_fuente = float(coincidencia.get("CantidadDisp", 0))
                else:
                    # Usar la función para calcular el disponible según la fuente y lote específico
                    disponible_fuente = obtener_disponible_por_fuente(
                        fuente=fuente,
                        material=material_solicitado,
                        centro=centro,
                        almacen=almacen,
                        df_fuente=df_fuente,
                        inventario_df=inventario_df,
                        lote=lote,
                    )

                # Fix 4: Omitir sugerencias sin disponible
                if disponible_fuente <= 0:
                    continue

                # En la función buscar_sugerencias_exactas (línea ~1580):
                fecha_cad = formatear_fecha_caducidad(fecha_cad)

                linea = crear_linea_sugerencia(
                    pedido=pedido,
                    material_sugerido=material_solicitado,
                    fuente=fuente,
                    centro_sugerido=centro,
                    almacen_sugerido=almacen,
                    disponible=disponible_fuente,
                    inventario_df=inventario_df,
                    lote=lote,
                    fecha_caducidad=fecha_cad,
                )
                sugerencias.append(linea)

    return sugerencias


# =========================
# NUEVA FUNCIÓN: Enriquecer Todas las Sugerencias con Meses_Inventario y Promedio_Consumo_12M
# tomados desde el Resumen Sin Sugerencias (clave: Centro/Material/Almacen)
# =========================
def enriquecer_sugerencias_con_consumo(
    df_sugerencias: pd.DataFrame,
    df_resumen: pd.DataFrame,
    df_facturacion: pd.DataFrame = None,
    df_reporte_consumo: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Post-proceso vectorizado que agrega a df_sugerencias:
      - Promedio_Consumo_12M              (consumo promedio Centro/Material/Almacen del Resumen)
      - Consumo promedio (Dest/Material)  (Consumo_promedio_mensual del Reporte de Consumo,
                                           lookup por Destinatario + Material solicitado)
      - Meses_Inventario                  (inventario del almacén del pedido / Promedio_Consumo_12M)
    Regla de inventario:
      Almacen 1030 → Inv 1030 | 1031 → Inv 1031 | 1060 → Inv 1060 | otro → Inv 1032
    """
    if df_sugerencias is None or df_sugerencias.empty:
        return df_sugerencias

    df = df_sugerencias.copy()

    # ── 1. Promedio_Consumo_12M desde Resumen (Centro/Material/Almacen) ──────
    if df_resumen is not None and not df_resumen.empty:
        cols_needed = ["Centro", "Material", "Almacen", "Promedio_Consumo_12M"]
        if all(c in df_resumen.columns for c in cols_needed):
            lookup_resumen = (
                df_resumen[cols_needed]
                .drop_duplicates(subset=["Centro", "Material", "Almacen"])
                .copy()
            )
            # Renombrar para evitar colisión con columnas del df principal
            lookup_resumen = lookup_resumen.rename(
                columns={
                    "Centro": "_r_centro",
                    "Material": "_r_material",
                    "Almacen": "_r_almacen",
                    "Promedio_Consumo_12M": "_prom_resumen",
                }
            )
            df = df.merge(
                lookup_resumen,
                left_on=[
                    Columnas.CENTRO_PEDIDO,
                    Columnas.MATERIAL_SOLICITADO,
                    Columnas.ALMACEN,
                ],
                right_on=["_r_centro", "_r_material", "_r_almacen"],
                how="left",
            )
            df[Columnas.PROMEDIO_CONSUMO_12M] = df["_prom_resumen"].fillna(0)
            # Limpiar columnas auxiliares
            df.drop(
                columns=[
                    c
                    for c in ["_r_centro", "_r_material", "_r_almacen", "_prom_resumen"]
                    if c in df.columns
                ],
                inplace=True,
            )
    else:
        df[Columnas.PROMEDIO_CONSUMO_12M] = df.get(
            Columnas.PROMEDIO_CONSUMO_12M, pd.Series(0, index=df.index)
        ).fillna(0)

    # ── 2. Consumo promedio (Destinatario/Material) desde Reporte de Consumo ──
    # Se usa clave concatenada (igual que un BUSCARV con columna auxiliar en Excel)
    # para evitar problemas de tipo int/str, espacios o sufijos .0 en el merge.
    def _normalizar_clave(serie: pd.Series) -> pd.Series:
        return (
            serie.astype(str)
            .str.strip()
            .str.replace(r"\.0+$", "", regex=True)
            .str.upper()
        )

    if df_reporte_consumo is not None and not df_reporte_consumo.empty:
        try:
            cols_lookup = ["Destinatario", "Material", "Consumo_promedio_mensual"]
            if all(c in df_reporte_consumo.columns for c in cols_lookup):
                lookup_rc = df_reporte_consumo[cols_lookup].drop_duplicates(
                    subset=["Destinatario", "Material"]
                ).copy()

                # Clave concatenada en el Reporte de Consumo
                lookup_rc["_rc_key"] = (
                    _normalizar_clave(lookup_rc["Destinatario"])
                    + "||"
                    + _normalizar_clave(lookup_rc["Material"])
                )
                lookup_rc = lookup_rc[["_rc_key", "Consumo_promedio_mensual"]].rename(
                    columns={"Consumo_promedio_mensual": "_rc_consumo"}
                )

                # Clave concatenada en Todas las Sugerencias
                df["_sug_key"] = (
                    _normalizar_clave(df[Columnas.DESTINATARIO])
                    + "||"
                    + _normalizar_clave(df[Columnas.MATERIAL_SOLICITADO])
                )

                df = df.merge(lookup_rc, left_on="_sug_key", right_on="_rc_key", how="left")
                df[Columnas.CONSUMO_DESTINATARIO_12M] = df["_rc_consumo"].fillna(0)
                df.drop(
                    columns=[c for c in ["_sug_key", "_rc_key", "_rc_consumo"] if c in df.columns],
                    inplace=True,
                )
            else:
                logger.warning(
                    f"Reporte de Consumo no tiene las columnas esperadas. "
                    f"Columnas disponibles: {df_reporte_consumo.columns.tolist()}"
                )
                df[Columnas.CONSUMO_DESTINATARIO_12M] = 0.0
        except Exception as e:
            logger.warning(f"No se pudo calcular Consumo promedio (Destinatario/Material): {e}")
            df[Columnas.CONSUMO_DESTINATARIO_12M] = 0.0
    else:
        df[Columnas.CONSUMO_DESTINATARIO_12M] = 0.0

    # ── Calcular Meses_Inventario con inventario por almacén del pedido ──
    almacen_col = df[Columnas.ALMACEN].astype(str).str.strip()
    inv_segun_almacen = np.select(
        [
            almacen_col == "1030",
            almacen_col == "1031",
            almacen_col == "1060",
        ],
        [
            pd.to_numeric(df[Columnas.INV_1030], errors="coerce").fillna(0),
            pd.to_numeric(df[Columnas.INV_1031], errors="coerce").fillna(0),
            pd.to_numeric(df[Columnas.INV_1060], errors="coerce").fillna(0),
        ],
        default=pd.to_numeric(df[Columnas.INV_1032], errors="coerce").fillna(0),
    )
    consumo_prom = pd.to_numeric(
        df[Columnas.PROMEDIO_CONSUMO_12M], errors="coerce"
    ).fillna(0)
    df[Columnas.MESES_INVENTARIO] = np.where(
        consumo_prom > 0,
        (inv_segun_almacen / consumo_prom).round(2),
        np.where(inv_segun_almacen == 0, 0.0, 999.0),
    )

    return df


# =========================
# Actualizar generar_todas_sugerencias (versión optimizada)
# =========================
def generar_todas_sugerencias(
    pedidos_df: pd.DataFrame,
    hojas_externas: Dict[str, pd.DataFrame],
    fuentes_activas: List[str],
    inventario_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera todas las sugerencias para todos los pedidos, incluyendo línea sin
    sugerencia.

    Optimización respecto a la versión anterior (row-by-row con escaneos O(n)):
      · FASE A – Pre-indexa inventario y fuentes UNA sola vez.
      · FASE B – Calcula templates de sugerencia por par único (Material,Centro):
                 reduce iteraciones pesadas de N_pedidos a N_unique_pares.
      · FASE C – Ensambla todas las líneas con lookups O(1) sin tocar DataFrames.

    Speedup típico: 30–150x dependiendo de la cardinalidad de pedidos.
    Los helpers _build_inv_caches_rc, _build_fuentes_index_rc,
    _buscar_templates_sug_rc y _montar_linea_pedido son compartidos con el
    motor de 'Sugerencias desde Reporte de Consumo'.
    """
    if pedidos_df is None or pedidos_df.empty:
        return pd.DataFrame()

    progress_bar = st.progress(0)
    status_text  = st.empty()

    # ── FASE A: Pre-indexado ─────────────────────────────────────────────
    status_text.text("⚙️ Pre-indexando inventario…")
    inv_caches = _build_inv_caches_rc(inventario_df)
    progress_bar.progress(0.10)

    status_text.text("⚙️ Pre-indexando fuentes externas…")
    idx_fuentes = _build_fuentes_index_rc(hojas_externas, fuentes_activas)
    progress_bar.progress(0.20)

    # ── FASE B: Templates por par único (Material, Centro) ────────────────
    pares_unicos = (
        pedidos_df[["Material", "Centro"]]
        .drop_duplicates()
        .dropna(subset=["Material"])
    )
    pares_unicos = pares_unicos[
        pares_unicos["Material"].astype(str).str.strip() != ""
    ]
    total_pares = max(len(pares_unicos), 1)

    status_text.text(
        f"🔎 Calculando sugerencias para {total_pares:,} pares únicos "
        f"(Material, Centro)…"
    )
    templates_cache: Dict[tuple, List[dict]] = {}
    for i, (_, pair) in enumerate(pares_unicos.iterrows()):
        mat = str(pair.get("Material", "") or "").strip()
        cen = str(pair.get("Centro",   "") or "").strip()
        if not mat:
            continue
        templates_cache[(mat, cen)] = _buscar_templates_sug_rc(
            mat, fuentes_activas, idx_fuentes, inv_caches
        )
        if i % max(1, total_pares // 50) == 0:
            progress_bar.progress(0.20 + 0.45 * (i / total_pares))

    progress_bar.progress(0.65)

    # ── FASE C: Ensamblar todas las líneas con O(1) lookups ───────────────
    total_pedidos = len(pedidos_df)
    status_text.text(f"📋 Ensamblando {total_pedidos:,} pedidos…")

    todas_sugerencias: List[dict] = []
    for i, (_, pedido) in enumerate(pedidos_df.iterrows()):
        mat = str(pedido.get("Material", "") or "").strip()
        cen = str(pedido.get("Centro",   "") or "").strip()
        if not mat:
            continue

        # Línea sin sugerencia
        todas_sugerencias.append(
            _montar_linea_pedido(pedido, None, inv_caches)
        )

        # Líneas con sugerencia (lookup O(1) en templates_cache)
        for tmpl in templates_cache.get((mat, cen), []):
            todas_sugerencias.append(
                _montar_linea_pedido(pedido, tmpl, inv_caches)
            )

        if i % max(1, total_pedidos // 50) == 0:
            progress_bar.progress(0.65 + 0.30 * (i / total_pedidos))

    progress_bar.empty()
    status_text.empty()

    if not todas_sugerencias:
        return pd.DataFrame()

    df_resultado = pd.DataFrame(todas_sugerencias)
    df_resultado = consolidar_sugerencias_repetidas(df_resultado)

    # Ordenar columnas en el orden exacto solicitado
    columnas_orden = [
        Columnas.GRUPO_CLIENTE,         Columnas.FECHA,
        Columnas.PEDIDO,                Columnas.GRUPO_VENDEDOR,
        Columnas.SOLICITANTE,           Columnas.DESTINATARIO,
        Columnas.RAZON_SOCIAL,          Columnas.CENTRO_PEDIDO,
        Columnas.ALMACEN,               Columnas.MATERIAL_SOLICITADO,
        Columnas.MATERIAL_BASE,         Columnas.DESCRIPCION_SOLICITADA,
        Columnas.CANTIDAD_PEDIDO,       Columnas.CANTIDAD_PENDIENTE,
        Columnas.CANTIDAD_OFERTAR,      Columnas.PRECIO,
        Columnas.CONSUMO_DESTINATARIO_12M,  # ← antes de Fuente
        Columnas.FUENTE,                Columnas.MATERIAL_SUGERIDO,
        Columnas.DESCRIPCION_SUGERIDA,  Columnas.CENTRO_SUGERIDO,
        Columnas.ALMACEN_SUGERIDO,      Columnas.DISPONIBLE,
        Columnas.LOTE,                  Columnas.FECHA_CADUCIDAD,
        Columnas.CENTRO_INV,            Columnas.INV_1030,
        Columnas.INV_1031,              Columnas.INV_1032,
        Columnas.INV_1060,              Columnas.MESES_INVENTARIO,
        Columnas.PROMEDIO_CONSUMO_12M,  Columnas.CANT_TRANSITO,
        Columnas.CANT_TRANSITO_1030,    Columnas.CANT_TRANSITO_1031,
        Columnas.CANT_TRANSITO_1032,    Columnas.DISP_1031_1030,
        Columnas.DISP_1031_1032,        Columnas.INV_1001,
        Columnas.INV_1003,              Columnas.INV_1004,
        Columnas.INV_1017,              Columnas.INV_1018,
        Columnas.INV_1022,              Columnas.INV_1036,
        Columnas.BLOQUEADO,
    ]

    for col in columnas_orden:
        if col not in df_resultado.columns:
            df_resultado[col] = ""

    return df_resultado[columnas_orden]


# =============================================================================
# OPTIMIZACIÓN: Motor compartido para "Todas las Sugerencias" y
#               "Sugerencias desde Reporte de Consumo"
#
# Helpers compartidos (definidos más abajo en el bloque RC):
#   _build_inv_caches_rc(inventario_df)            → pre-indexa inventario
#   _build_fuentes_index_rc(hojas_externas, ...)   → pre-indexa fuentes
#   _buscar_templates_sug_rc(material, ...)        → templates por par único
#
# Helper específico de pedidos (abajo):
#   _montar_linea_pedido(pedido, template, caches) → ensambla línea con campos
#                                                    reales del pedido (número,
#                                                    almacén, bloqueado, etc.)
# =============================================================================

def _montar_linea_pedido(
    pedido: pd.Series,
    template: Optional[dict],
    inv_caches: dict,
) -> dict:
    """
    Ensambla una línea de salida para 'Todas las Sugerencias' usando:
      · pedido   — pd.Series con los campos reales del DF de pedidos
      · template — dict producido por _buscar_templates_sug_rc, o None
                   (None = línea sin sugerencia)
      · inv_caches — dict pre-construido por _build_inv_caches_rc

    Todos los lookups de inventario son O(1) contra los caches; no se toca
    ningún DataFrame.  Los campos exclusivos de pedidos que no existen en el
    Reporte de Consumo (Pedido, Almacén real, Sts. Créd., Bloqueo Ent.) se
    leen directamente de la serie, igual que hacía crear_linea_sin_sugerencia /
    crear_linea_sugerencia pero sin los escaneos a inventario_df.
    """
    centro   = str(pedido.get("Centro",   "") or "").strip()
    material = str(pedido.get("Material", "") or "").strip()

    if template is not None:
        mat_inv    = template["material_inv_key"]
        fuente_val = template["fuente"]
        mat_sug    = template["material_sugerido"]
        desc_sug   = template["descripcion_sugerida"]
        centro_sug = template["centro_sugerido"]
        alm_sug    = template["almacen_sugerido"]
        disponible = template["disponible"]
        lote_val   = template["lote"]
        fec_cad    = template["fecha_caducidad"]
    else:
        mat_inv    = material
        fuente_val = ""
        mat_sug    = ""
        desc_sug   = ""
        centro_sug = ""
        alm_sug    = ""
        disponible = 0.0
        lote_val   = ""
        fec_cad    = ""

    # ── Inventario O(1) ───────────────────────────────────────────────────
    inv_alm  = inv_caches["inv_centro_alm"].get((centro, mat_inv), {})
    inv_1030 = inv_alm.get("1030", 0.0)
    inv_1031 = inv_alm.get("1031", 0.0)
    inv_1032 = inv_alm.get("1032", 0.0)
    inv_1060 = inv_alm.get("1060", 0.0)

    tr        = inv_caches["transito"].get((centro, mat_inv), {})
    tr_total  = sum(tr.values())

    disp_1031_1030 = inv_caches["disp_1031"].get((mat_inv, "1030"), 0.0)
    disp_1031_1032 = inv_caches["disp_1031"].get((mat_inv, "1032"), 0.0)

    inv_por_centro = inv_caches["inv_filtrado_mat"].get(mat_inv, {})

    # ── Campos exclusivos de pedidos ──────────────────────────────────────
    pendiente = float(pedido.get("Pendiente", 0) or 0)
    cantidad_ofertar = (
        min(pendiente, disponible)
        if (template is not None and pendiente > 0)
        else 0.0
    )

    bloqueado_val = ""
    if str(pedido.get("Sts. Créd.", "") or "").strip() == "B":
        bloqueado_val = "Crédito"
    bloqueo_ent = str(pedido.get("Bloqueo Ent.", "") or "").strip()
    if bloqueo_ent not in ("", "nan"):
        bloqueado_val = "Detenido por ambos" if bloqueado_val else "Detenido"

    return {
        Columnas.GRUPO_CLIENTE:          str(pedido.get("Gpo. Cte.",     "") or "").strip(),
        Columnas.FECHA:                  pedido.get("Fecha",              ""),
        Columnas.PEDIDO:                 pedido.get("Pedido",             ""),
        Columnas.GRUPO_VENDEDOR:         pedido.get("Gpo.Vdor.",          ""),
        Columnas.SOLICITANTE:            pedido.get("Solicitante",         ""),
        Columnas.DESTINATARIO:           pedido.get("Destinatario",        ""),
        Columnas.RAZON_SOCIAL:           str(pedido.get("Razón Social",   "") or ""),
        Columnas.CENTRO_PEDIDO:          centro,
        Columnas.ALMACEN:                str(pedido.get("Almacén",        "") or "").strip(),
        Columnas.MATERIAL_SOLICITADO:    material,
        Columnas.MATERIAL_BASE:          material,
        Columnas.DESCRIPCION_SOLICITADA: str(pedido.get("Texto Material", "") or ""),
        Columnas.CANTIDAD_PEDIDO:        pedido.get("Cantidad",           ""),
        Columnas.CANTIDAD_PENDIENTE:     pendiente,
        Columnas.CANTIDAD_OFERTAR:       cantidad_ofertar,
        Columnas.PRECIO:                 pedido.get("Precio",             0),
        Columnas.FUENTE:                 fuente_val,
        Columnas.MATERIAL_SUGERIDO:      mat_sug,
        Columnas.DESCRIPCION_SUGERIDA:   desc_sug,
        Columnas.CENTRO_SUGERIDO:        centro_sug,
        Columnas.ALMACEN_SUGERIDO:       alm_sug,
        Columnas.DISPONIBLE:             disponible,
        Columnas.LOTE:                   lote_val,
        Columnas.FECHA_CADUCIDAD:        fec_cad,
        Columnas.CENTRO_INV:             centro,
        Columnas.INV_1030:               inv_1030,
        Columnas.INV_1031:               inv_1031,
        Columnas.INV_1032:               inv_1032,
        Columnas.INV_1060:               inv_1060,
        Columnas.MESES_INVENTARIO:       0.0,   # post-proceso enriquecer_sugerencias_con_consumo
        Columnas.PROMEDIO_CONSUMO_12M:   0.0,   # post-proceso
        Columnas.CONSUMO_DESTINATARIO_12M: 0.0, # post-proceso
        Columnas.CANT_TRANSITO:          tr_total,
        Columnas.CANT_TRANSITO_1030:     tr.get("1030", 0.0),
        Columnas.CANT_TRANSITO_1031:     tr.get("1031", 0.0),
        Columnas.CANT_TRANSITO_1032:     tr.get("1032", 0.0),
        Columnas.DISP_1031_1030:         disp_1031_1030,
        Columnas.DISP_1031_1032:         disp_1031_1032,
        Columnas.INV_1001:               inv_por_centro.get("1001", 0.0),
        Columnas.INV_1003:               inv_por_centro.get("1003", 0.0),
        Columnas.INV_1004:               inv_por_centro.get("1004", 0.0),
        Columnas.INV_1017:               inv_por_centro.get("1017", 0.0),
        Columnas.INV_1018:               inv_por_centro.get("1018", 0.0),
        Columnas.INV_1022:               inv_por_centro.get("1022", 0.0),
        Columnas.INV_1036:               inv_por_centro.get("1036", 0.0),
        Columnas.BLOQUEADO:              bloqueado_val,
    }


# =============================================================================
# OPTIMIZACIÓN: Motor de "Sugerencias desde Reporte de Consumo"
#
# Problema anterior: loop de 70k filas, cada una con 5-10 escaneos completos
# del DF de inventario (O(n) cada uno) + escaneos de fuentes externas.
#
# Solución en 3 fases:
#   A) Pre-indexar inventario y fuentes externas UNA SOLA VEZ → O(1) lookups
#   B) Calcular templates de sugerencia por par único (Material, Centro)
#      → reduce iteraciones pesadas de 70k a N_unique (típicamente 500-5k)
#   C) Armar las 70k líneas de salida con lookups O(1) (sin escanear DFs)
#
# Speedup esperado: 30-100x según cardinalidad del RC.
# =============================================================================

def _build_inv_caches_rc(inventario_df: pd.DataFrame) -> dict:
    """
    FASE A — Construye todos los dicts de inventario necesarios para el motor
    optimizado. Ejecuta una sola vez; reemplaza todas las llamadas a
    obtener_inventario_*, get_transito_* dentro del loop de 70k filas.

    Estructura devuelta:
      inv_centro_alm   : (centro, material) → {almacen: libre_utilizacion}
      transito         : (centro, material) → {almacen: cant_transito}
      inv_filtrado_mat : material → {centro: suma_libre_alm_1030/1031/1060}
      disp_1031        : (material, almacen) → libre  cuando centro == "1031"
    """
    caches: dict = {
        "inv_centro_alm":   {},   # (centro, mat)       → {alm: libre}
        "transito":         {},   # (centro, mat)       → {alm: transito}
        "inv_filtrado_mat": {},   # mat                 → {centro: suma_filt}
        "disp_1031":        {},   # (mat, alm)          → libre en centro 1031
    }

    if inventario_df is None or inventario_df.empty:
        return caches

    _inv = inventario_df.copy()
    _inv["Centro"]            = _inv["Centro"].astype(str).str.strip()
    _inv["Material"]          = _inv["Material"].astype(str).str.strip()
    _inv["Almacén"]           = _inv["Almacén"].astype(str).str.strip()
    _inv["Libre Utilización"] = pd.to_numeric(
        _inv.get("Libre Utilización", 0), errors="coerce"
    ).fillna(0.0)
    _inv["Cant. en Tránsito"] = pd.to_numeric(
        _inv.get("Cant. en Tránsito", 0), errors="coerce"
    ).fillna(0.0)

    # ── inv_centro_alm y transito ─────────────────────────────────────────
    grp = _inv.groupby(["Centro", "Material", "Almacén"], sort=False)
    for (centro, mat, alm), g in grp:
        key_cm = (centro, mat)
        libre  = float(g["Libre Utilización"].sum())
        trans  = float(g["Cant. en Tránsito"].sum())

        if key_cm not in caches["inv_centro_alm"]:
            caches["inv_centro_alm"][key_cm] = {}
        caches["inv_centro_alm"][key_cm][alm] = libre

        if key_cm not in caches["transito"]:
            caches["transito"][key_cm] = {}
        caches["transito"][key_cm][alm] = trans

    # ── inv_filtrado_mat (solo almacenes 1030 / 1031 / 1060) ─────────────
    _filt = _inv[_inv["Almacén"].isin(["1030", "1031", "1060"])]
    for (mat, centro), g in _filt.groupby(["Material", "Centro"], sort=False):
        if mat not in caches["inv_filtrado_mat"]:
            caches["inv_filtrado_mat"][mat] = {}
        caches["inv_filtrado_mat"][mat][centro] = float(g["Libre Utilización"].sum())

    # ── disp_1031 (libre en centro "1031" por material y almacén) ─────────
    _c1031 = _inv[_inv["Centro"] == "1031"]
    for (mat, alm), g in _c1031.groupby(["Material", "Almacén"], sort=False):
        caches["disp_1031"][(mat, alm)] = float(g["Libre Utilización"].sum())

    return caches


def _build_fuentes_index_rc(
    hojas_externas: Dict[str, pd.DataFrame],
    fuentes_activas: List[str],
) -> Dict[str, Dict[str, List[dict]]]:
    """
    FASE A — Convierte cada hoja externa en un índice {material: [row_dicts]}.
    Reemplaza df_fuente[df_fuente["Material"] == mat] por un O(1) dict.get().
    """
    idx: Dict[str, Dict[str, List[dict]]] = {}
    for fuente in fuentes_activas:
        if fuente not in hojas_externas:
            continue
        df_f = hojas_externas[fuente]
        if df_f.empty or "Material" not in df_f.columns:
            continue
        _df = df_f.copy()
        _df["Material"] = _df["Material"].astype(str).str.strip()
        idx[fuente] = {}
        for mat, grp in _df.groupby("Material", sort=False):
            idx[fuente][mat] = grp.to_dict("records")
    return idx


def _buscar_templates_sug_rc(
    material: str,
    fuentes_activas: List[str],
    idx_fuentes: Dict[str, Dict[str, List[dict]]],
    inv_caches: dict,
) -> List[dict]:
    """
    FASE B — Reproduce la lógica completa de buscar_sugerencias_exactas pero
    usando índices O(1) en lugar de escaneos de DataFrame.

    Devuelve una lista de "templates": campos de sugerencia que NO dependen de
    la fila específica del RC (fuente, mat_sugerido, centro_sugerido, etc.).
    El campo extra "material_inv_key" indica qué material usar al resolver las
    columnas de inventario en _montar_linea_rc.
    """
    templates: List[dict] = []
    if not material:
        return templates

    for fuente in fuentes_activas:
        if fuente not in idx_fuentes:
            continue

        # ── SUSTITUTO ────────────────────────────────────────────────────
        if fuente == "Sustituto":
            for s_row in idx_fuentes[fuente].get(material, []):
                mat_sust  = str(s_row.get("Material sustituto", "") or "").strip()
                desc_sust = str(s_row.get("Texto material sustituto", "") or "")
                if not mat_sust:
                    continue

                otras = [f for f in fuentes_activas if f not in ("Sustituto", "Lento mov")]
                encontrado = False
                for otra in otras:
                    for om in idx_fuentes.get(otra, {}).get(mat_sust, []):
                        disp = float(om.get("CantidadDisp", 0) or 0)
                        if disp <= 0:
                            continue
                        encontrado = True
                        templates.append({
                            "fuente":             f"Sustituto/{otra}",
                            "material_sugerido":  mat_sust,
                            "descripcion_sugerida": desc_sust,
                            "centro_sugerido":    str(om.get("Centro", "") or "").strip(),
                            "almacen_sugerido":   str(om.get("Almacén", "") or "").strip(),
                            "disponible":         disp,
                            "lote":               str(om.get("Lote", "") or "").strip(),
                            "fecha_caducidad":    formatear_fecha_caducidad(
                                                      om.get("FechaCaducidad", "")
                                                  ),
                            "material_inv_key":   mat_sust,
                        })

                if not encontrado:
                    disp_sust = sum(
                        inv_caches["inv_filtrado_mat"].get(mat_sust, {}).values()
                    )
                    if disp_sust > 0:
                        templates.append({
                            "fuente":             "Sustituto",
                            "material_sugerido":  mat_sust,
                            "descripcion_sugerida": desc_sust,
                            "centro_sugerido":    "",
                            "almacen_sugerido":   "",
                            "disponible":         disp_sust,
                            "lote":               "",
                            "fecha_caducidad":    "",
                            "material_inv_key":   mat_sust,
                        })

        # ── LENTO MOV ────────────────────────────────────────────────────
        elif fuente == "Lento mov":
            if not idx_fuentes.get(fuente, {}).get(material):
                continue

            otras     = [f for f in fuentes_activas if f not in ("Sustituto", "Lento mov")]
            encontrado = False
            for otra in otras:
                if encontrado:
                    break
                added_any = False
                for om in idx_fuentes.get(otra, {}).get(material, []):
                    disp = float(om.get("CantidadDisp", 0) or 0)
                    if disp <= 0:
                        continue
                    added_any = True
                    templates.append({
                        "fuente":             f"Lento mov/{otra}",
                        "material_sugerido":  material,
                        "descripcion_sugerida": "",
                        "centro_sugerido":    str(om.get("Centro", "") or "").strip(),
                        "almacen_sugerido":   str(om.get("Almacén", "") or "").strip(),
                        "disponible":         disp,
                        "lote":               str(om.get("Lote", "") or "").strip(),
                        "fecha_caducidad":    formatear_fecha_caducidad(
                                                  om.get("FechaCaducidad", "")
                                              ),
                        "material_inv_key":   material,
                    })
                if added_any:
                    encontrado = True  # avanza al siguiente 'otra' solo si ya se encontró algo

            if not encontrado:
                disp_lm = sum(
                    inv_caches["inv_filtrado_mat"].get(material, {}).values()
                )
                if disp_lm > 0:
                    templates.append({
                        "fuente":             "Lento mov",
                        "material_sugerido":  material,
                        "descripcion_sugerida": "",
                        "centro_sugerido":    "",
                        "almacen_sugerido":   "",
                        "disponible":         disp_lm,
                        "lote":               "",
                        "fecha_caducidad":    "",
                        "material_inv_key":   material,
                    })

        # ── CORTA CADUCIDAD / COSMOPARK / PNC / CADUCO ───────────────────
        else:
            for match in idx_fuentes.get(fuente, {}).get(material, []):
                disp = float(match.get("CantidadDisp", 0) or 0)
                if disp <= 0:
                    continue
                templates.append({
                    "fuente":             fuente,
                    "material_sugerido":  material,
                    "descripcion_sugerida": str(match.get("Descripcion", "") or ""),
                    "centro_sugerido":    str(match.get("Centro", "") or "").strip(),
                    "almacen_sugerido":   str(match.get("Almacén", "") or "").strip(),
                    "disponible":         disp,
                    "lote":               str(match.get("Lote", "") or "").strip(),
                    "fecha_caducidad":    formatear_fecha_caducidad(
                                              match.get("FechaCaducidad", "")
                                          ),
                    "material_inv_key":   material,
                })

    return templates


def _montar_linea_rc(
    pedido_fields: dict,
    template: Optional[dict],
    inv_caches: dict,
    rc_row_all: Optional[dict] = None,
) -> dict:
    """
    FASE C — Combina los campos del RC con un template de sugerencia (o None
    para línea sin sugerencia). Todos los lookups de inventario son O(1).

    Cuando se pasa rc_row_all (row.to_dict() de df_reporte_consumo), el dict
    resultante incluye TODAS las columnas originales del Reporte de Consumo
    con sus nombres exactos, seguidas de las columnas de sugerencias.

    Columnas internas que NO aparecen en el output final pero son necesarias
    para que enriquecer_sugerencias_con_consumo funcione correctamente:
      Columnas.CENTRO_PEDIDO      → "Centro pedido"   (alias de "Centro")
      Columnas.MATERIAL_SOLICITADO → "Material solicitado" (alias de "Material")
      Columnas.ALMACEN            → "Almacén"          (vacío en RC)
    """
    centro   = pedido_fields["centro"]
    material = pedido_fields["material"]
    rc       = rc_row_all or {}

    if template is not None:
        mat_inv    = template["material_inv_key"]
        fuente_val = template["fuente"]
        mat_sug    = template["material_sugerido"]
        desc_sug   = template["descripcion_sugerida"]
        centro_sug = template["centro_sugerido"]
        alm_sug    = template["almacen_sugerido"]
        disponible = template["disponible"]
        lote_val   = template["lote"]
        fec_cad    = template["fecha_caducidad"]
    else:
        mat_inv    = material
        fuente_val = ""
        mat_sug    = ""
        desc_sug   = ""
        centro_sug = ""
        alm_sug    = ""
        disponible = 0.0
        lote_val   = ""
        fec_cad    = ""

    # ── Inventario O(1) ───────────────────────────────────────────────────
    inv_alm  = inv_caches["inv_centro_alm"].get((centro, mat_inv), {})
    inv_1030 = inv_alm.get("1030", 0.0)
    inv_1031 = inv_alm.get("1031", 0.0)
    inv_1032 = inv_alm.get("1032", 0.0)
    inv_1060 = inv_alm.get("1060", 0.0)

    tr       = inv_caches["transito"].get((centro, mat_inv), {})
    tr_total = sum(tr.values())

    disp_1031_1030 = inv_caches["disp_1031"].get((mat_inv, "1030"), 0.0)
    disp_1031_1032 = inv_caches["disp_1031"].get((mat_inv, "1032"), 0.0)

    inv_por_centro = inv_caches["inv_filtrado_mat"].get(mat_inv, {})

    pendiente        = pedido_fields["pendiente"]
    cantidad_ofertar = (
        min(pendiente, disponible) if (template is not None and pendiente > 0) else 0.0
    )

    def _s(key, default=""):
        return str(rc.get(key, default) or default).strip()

    def _f(key, default=0.0):
        try:
            return float(rc.get(key, default) or default)
        except (ValueError, TypeError):
            return float(default)

    return {
        # ── Columnas originales del Reporte de Consumo (nombres exactos) ──
        "Centro":                          centro,
        "Grp. Cliente":                    _s("Grp. Cliente"),
        "Gpo. Vdor.":                      _s("Gpo. Vdor."),
        "Solicitante":                     _s("Solicitante"),
        "Destinatario":                    _s("Destinatario"),
        "Razón Social":                    _s("Razón Social"),
        "Material":                        material,
        "Texto Material":                  _s("Texto Material"),
        "Ultima_compra_cliente":           _s("Ultima_compra_cliente"),
        "Ultima_facturacion_destinatario": _s("Ultima_facturacion_destinatario"),
        "Consumo_promedio_mensual":        pedido_fields["pendiente"],
        "Consumo_actual":                  _f("Consumo_actual"),
        "UM":                              _s("UM"),
        "Tendencia":                       _f("Tendencia"),
        "Tendencia de cantidad":           _f("Tendencia de cantidad"),
        "Ultimo mes facturacion":          _s("Ultimo mes facturacion"),
        "Cantidad ultima":                 pedido_fields["cantidad"],
        "Importe ultima":                  _f("Importe ultima"),
        "Precio_unitario_ultima":          pedido_fields["precio"],
        "Penultima_fecha":                 _s("Penultima_fecha"),
        "Cantidad_penultima":              _f("Cantidad_penultima"),
        "Importe_penultima":               _f("Importe_penultima"),
        "Precio_unitario_penultima":       _f("Precio_unitario_penultima"),
        "precio_min":                      _f("precio_min"),
        "precio_max":                      _f("precio_max"),
        "precio_prom":                     _f("precio_prom"),
        # ── Columnas de sugerencia (Columnas.* constants) ──────────────────
        Columnas.FUENTE:                   fuente_val,
        Columnas.MATERIAL_SUGERIDO:        mat_sug,
        Columnas.DESCRIPCION_SUGERIDA:     desc_sug,
        Columnas.CENTRO_SUGERIDO:          centro_sug,
        Columnas.ALMACEN_SUGERIDO:         alm_sug,
        Columnas.DISPONIBLE:               disponible,
        Columnas.LOTE:                     lote_val,
        Columnas.FECHA_CADUCIDAD:          fec_cad,
        Columnas.CENTRO_INV:               centro,
        Columnas.INV_1030:                 inv_1030,
        Columnas.INV_1031:                 inv_1031,
        Columnas.INV_1032:                 inv_1032,
        Columnas.INV_1060:                 inv_1060,
        Columnas.MESES_INVENTARIO:         0.0,   # post-proceso
        Columnas.PROMEDIO_CONSUMO_12M:     0.0,   # post-proceso
        Columnas.CONSUMO_DESTINATARIO_12M: 0.0,   # post-proceso
        Columnas.CANT_TRANSITO:            tr_total,
        Columnas.CANT_TRANSITO_1030:       tr.get("1030", 0.0),
        Columnas.CANT_TRANSITO_1031:       tr.get("1031", 0.0),
        Columnas.CANT_TRANSITO_1032:       tr.get("1032", 0.0),
        Columnas.DISP_1031_1030:           disp_1031_1030,
        Columnas.DISP_1031_1032:           disp_1031_1032,
        Columnas.INV_1001:                 inv_por_centro.get("1001", 0.0),
        Columnas.INV_1003:                 inv_por_centro.get("1003", 0.0),
        Columnas.INV_1004:                 inv_por_centro.get("1004", 0.0),
        Columnas.INV_1017:                 inv_por_centro.get("1017", 0.0),
        Columnas.INV_1018:                 inv_por_centro.get("1018", 0.0),
        Columnas.INV_1022:                 inv_por_centro.get("1022", 0.0),
        Columnas.INV_1036:                 inv_por_centro.get("1036", 0.0),
        # ── Alias internos para enriquecer_sugerencias_con_consumo ─────────
        # (no aparecen en columnas_orden → se excluyen del output final)
        Columnas.CENTRO_PEDIDO:            centro,    # "Centro pedido"
        Columnas.MATERIAL_SOLICITADO:      material,  # "Material solicitado"
        Columnas.ALMACEN:                  "",        # "Almacén"
    }


# =========================
# NUEVA FUNCIÓN: Generar sugerencias usando como base el Reporte de Consumo
# (versión optimizada — ver sección de pre-indexado arriba)
# =========================
def generar_sugerencias_desde_reporte_consumo(
    df_reporte_consumo: pd.DataFrame,
    hojas_externas: Dict[str, pd.DataFrame],
    fuentes_activas: List[str],
    inventario_df: pd.DataFrame,
    df_resumen: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Genera un reporte de sugerencias tomando como base cada fila del Reporte
    de Consumo. Aplica la misma lógica que 'Todas las Sugerencias' pero sobre
    registros históricos de facturación.

    Optimización respecto a la versión ingénua (row-by-row con escaneos):
      · FASE A – Pre-indexa inventario y fuentes una sola vez (O(n) total).
      · FASE B – Calcula templates de sugerencia por par único (Material, Centro):
                 reduce iteraciones pesadas de ~70 k a N_unique pares.
      · FASE C – Arma las 70 k líneas con lookups O(1); sin escanear DataFrames.

    Mapeo de columnas RC → pedido interno documentado en _montar_linea_rc.
    """
    if df_reporte_consumo is None or df_reporte_consumo.empty:
        return pd.DataFrame()

    progress_bar = st.progress(0)
    status_text  = st.empty()

    # ── FASE A: Pre-indexado (ejecuta una sola vez) ───────────────────────
    status_text.text("⚙️ Pre-indexando inventario…")
    inv_caches = _build_inv_caches_rc(inventario_df)
    progress_bar.progress(0.10)

    status_text.text("⚙️ Pre-indexando fuentes externas…")
    idx_fuentes = _build_fuentes_index_rc(hojas_externas, fuentes_activas)
    progress_bar.progress(0.20)

    # ── FASE B: Templates por par único (Material, Centro) ────────────────
    pares_unicos = (
        df_reporte_consumo[["Material", "Centro"]]
        .drop_duplicates()
        .dropna(subset=["Material"])
    )
    pares_unicos = pares_unicos[
        pares_unicos["Material"].astype(str).str.strip() != ""
    ]

    total_pares   = max(len(pares_unicos), 1)
    templates_cache: Dict[tuple, List[dict]] = {}

    status_text.text(
        f"🔎 Calculando sugerencias para {total_pares:,} pares únicos (Material, Centro)…"
    )
    for i, (_, pair) in enumerate(pares_unicos.iterrows()):
        material = str(pair.get("Material", "") or "").strip()
        centro   = str(pair.get("Centro", "") or "").strip()
        if not material:
            continue
        templates_cache[(material, centro)] = _buscar_templates_sug_rc(
            material, fuentes_activas, idx_fuentes, inv_caches
        )
        if i % max(1, total_pares // 50) == 0:
            progress_bar.progress(0.20 + 0.45 * (i / total_pares))

    progress_bar.progress(0.65)

    # ── FASE C: Armar todas las líneas de salida (O(1) por fila) ─────────
    total_rows = len(df_reporte_consumo)
    status_text.text(f"📋 Armando {total_rows:,} líneas de reporte…")

    todas_lineas: List[dict] = []
    for i, (_, row) in enumerate(df_reporte_consumo.iterrows()):
        material = str(row.get("Material", "") or "").strip()
        centro   = str(row.get("Centro", "") or "").strip()
        if not material:
            continue

        pedido_fields = {
            "gpo_cte":       str(row.get("Grp. Cliente",            "") or "").strip(),
            "fecha":         str(row.get("Ultima_compra_cliente",    "") or "").strip(),
            "gpo_vdor":      str(row.get("Gpo. Vdor.",              "") or "").strip(),
            "solicitante":   str(row.get("Solicitante",              "") or "").strip(),
            "destinatario":  str(row.get("Destinatario",             "") or "").strip(),
            "razon_social":  str(row.get("Razón Social",             "") or "").strip(),
            "centro":        centro,
            "material":      material,
            "texto_material": str(row.get("Texto Material",          "") or "").strip(),
            "cantidad":      float(row.get("Cantidad ultima",         0)  or 0),
            "pendiente":     float(row.get("Consumo_promedio_mensual", 0) or 0),
            "precio":        float(row.get("Precio_unitario_ultima",   0) or 0),
        }

        # Línea sin sugerencia
        todas_lineas.append(_montar_linea_rc(pedido_fields, None, inv_caches, row.to_dict()))

        # Líneas con sugerencia (templates pre-calculados → O(1) lookup)
        for tmpl in templates_cache.get((material, centro), []):
            todas_lineas.append(_montar_linea_rc(pedido_fields, tmpl, inv_caches, row.to_dict()))

        if i % max(1, total_rows // 50) == 0:
            progress_bar.progress(0.65 + 0.25 * (i / total_rows))

    progress_bar.progress(0.90)
    status_text.text("✔️ Consolidando y enriqueciendo…")

    progress_bar.empty()
    status_text.empty()

    if not todas_lineas:
        return pd.DataFrame()

    df_resultado = pd.DataFrame(todas_lineas)
    df_resultado = consolidar_sugerencias_repetidas(df_resultado)
    df_resultado = enriquecer_sugerencias_con_consumo(
        df_resultado,
        df_resumen if df_resumen is not None else pd.DataFrame(),
        df_reporte_consumo=df_reporte_consumo,
    )

    # ── Orden de columnas EXACTO (54 columnas, nombres literales) ────────
    # Grupo 1: columnas del Reporte de Consumo (26 cols)
    # Grupo 2: columnas de sugerencias         (28 cols)
    # Columnas internas de alias usadas por enriquecer_sugerencias_con_consumo
    # ("Centro pedido", "Material solicitado", "Almacén") quedan fuera por no
    # estar listadas aquí.
    COLUMNAS_FINALES_RC = [
        # ── Grupo 1: Reporte de Consumo ───────────────────────────────────
        "Centro",
        "Grp. Cliente",
        "Gpo. Vdor.",
        "Solicitante",
        "Destinatario",
        "Razón Social",
        "Material",
        "Texto Material",
        "Ultima_compra_cliente",
        "Ultima_facturacion_destinatario",
        "Consumo_promedio_mensual",
        "Consumo_actual",
        "UM",
        "Tendencia",
        "Tendencia de cantidad",
        "Ultimo mes facturacion",
        "Cantidad ultima",
        "Importe ultima",
        "Precio_unitario_ultima",
        "Penultima_fecha",
        "Cantidad_penultima",
        "Importe_penultima",
        "Precio_unitario_penultima",
        "precio_min",
        "precio_max",
        "precio_prom",
        # ── Grupo 2: Sugerencias ──────────────────────────────────────────
        "Fuente",
        "Material sugerido",
        "Descripción sugerida",
        "Centro sugerido",
        "Almacén sugerido",
        "Disponible",
        "Lote",
        "Fecha de Caducidad",
        "Centro (Inv)",
        "Inv 1030",
        "Inv 1031",
        "Inv 1032",
        "Inv 1060",
        "Meses_Inventario",
        "Promedio_Consumo_12M",
        "Cant. en Tránsito",
        "Cant. en Tránsito 1030",
        "Cant. en Tránsito 1031",
        "Cant. en Tránsito 1032",
        "Disponible 1031-1030",
        "Disponible 1031-1032",
        "Inv 1001",
        "Inv 1003",
        "Inv 1004",
        "Inv 1017",
        "Inv 1018",
        "Inv 1022",
        "Inv 1036",
    ]

    # Garantizar que cada columna exista (sin lanzar KeyError)
    for col in COLUMNAS_FINALES_RC:
        if col not in df_resultado.columns:
            df_resultado[col] = ""

    return df_resultado[COLUMNAS_FINALES_RC]


# =========================
# NUEVA FUNCIÓN: Calcular estadísticas de consumo por Centro/Material/Almacén
# =========================
def calcular_estadisticas_consumo_por_centro_material_almacen(
    df_facturacion_procesado: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula estadísticas de consumo por Centro/Material/Almacén:
    - Promedio consumo últimos 12 meses
    - Último mes de consumo (MM/AAAA)
    - Penúltimo mes de consumo (MM/AAAA)
    - Cantidad facturada último mes
    - Cantidad facturada penúltimo mes
    """
    if df_facturacion_procesado.empty:
        return pd.DataFrame()

    try:
        # Asegurar columnas necesarias
        columnas_necesarias = ["Centro", "Material", "Almacén", "Fecha", "Cantidad"]
        for col in columnas_necesarias:
            if col not in df_facturacion_procesado.columns:
                logger.warning(f"Columna {col} no encontrada en datos de facturación")
                return pd.DataFrame()

        # Convertir fecha
        df_facturacion_procesado["Fecha"] = pd.to_datetime(
            df_facturacion_procesado["Fecha"], errors="coerce"
        )

        # Crear columna de mes-año (MM/AAAA) para display
        df_facturacion_procesado["MesAno_str"] = df_facturacion_procesado[
            "Fecha"
        ].dt.strftime("%m/%Y")

        # Crear columna de mes-año numérica para ordenamiento (YYYYMM)
        df_facturacion_procesado["MesAno_num"] = (
            df_facturacion_procesado["Fecha"].dt.year * 100
            + df_facturacion_procesado["Fecha"].dt.month
        )

        # Filtrar solo fechas válidas; cantidades negativas (devoluciones) se incluyen
        # para reflejar el consumo neto real
        df_valido = df_facturacion_procesado[
            df_facturacion_procesado["Fecha"].notna()
        ].copy()

        if df_valido.empty:
            return pd.DataFrame()

        # ── Totales netos por Centro/Material/Almacén/Mes ────────────────────
        monthly = (
            df_valido.groupby(
                ["Centro", "Material", "Almacén", "MesAno_num", "MesAno_str"]
            )["Cantidad"]
            .sum()
            .reset_index()
        )

        KEYS = ["Centro", "Material", "Almacén"]

        # ── Últimos 12 meses desde la fecha máxima global ────────────────────
        fecha_maxima = df_valido["Fecha"].max()
        fecha_inicio_12m = fecha_maxima - pd.DateOffset(months=12)
        mes_inicio_12m_num = fecha_inicio_12m.year * 100 + fecha_inicio_12m.month

        monthly_12m = monthly[monthly["MesAno_num"] >= mes_inicio_12m_num]

        # Promedio 12M: suma neta / número de meses únicos en el período
        promedio_df = (
            monthly_12m.groupby(KEYS)
            .agg(
                total_12m=("Cantidad", "sum"),
                meses_12m=("MesAno_num", "nunique"),
            )
            .reset_index()
        )
        promedio_df["Promedio_Consumo_12M"] = (
            (promedio_df["total_12m"] / promedio_df["meses_12m"]).fillna(0).round(2)
        )

        # ── Último y penúltimo mes por grupo ─────────────────────────────────
        monthly_sorted = monthly.sort_values(
            KEYS + ["MesAno_num"], ascending=[True, True, True, False]
        )
        monthly_sorted["rank"] = monthly_sorted.groupby(KEYS).cumcount() + 1

        ult = monthly_sorted[monthly_sorted["rank"] == 1][
            KEYS + ["MesAno_str", "Cantidad"]
        ].rename(
            columns={
                "MesAno_str": "Ultimo_Mes_Consumo",
                "Cantidad": "Cantidad_Ultimo_Mes",
            }
        )
        pen = monthly_sorted[monthly_sorted["rank"] == 2][
            KEYS + ["MesAno_str", "Cantidad"]
        ].rename(
            columns={
                "MesAno_str": "Penultimo_Mes_Consumo",
                "Cantidad": "Cantidad_Penultimo_Mes",
            }
        )

        # ── Combinar todo ─────────────────────────────────────────────────────
        result = promedio_df[KEYS + ["Promedio_Consumo_12M"]].copy()
        result = result.merge(ult, on=KEYS, how="left")
        result = result.merge(pen, on=KEYS, how="left")

        result["Ultimo_Mes_Consumo"] = result["Ultimo_Mes_Consumo"].fillna("")
        result["Penultimo_Mes_Consumo"] = result["Penultimo_Mes_Consumo"].fillna("")
        result["Cantidad_Ultimo_Mes"] = result["Cantidad_Ultimo_Mes"].fillna(0)
        result["Cantidad_Penultimo_Mes"] = result["Cantidad_Penultimo_Mes"].fillna(0)

        # Renombrar Almacén → Almacen para consistencia con el resto del código
        result = result.rename(columns={"Almacén": "Almacen"})

        return result

    except Exception as e:
        logger.error(f"Error al calcular estadísticas de consumo: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return pd.DataFrame()


def validar_coherencia_temporal(df_resumen: pd.DataFrame):
    """
    Valida que el último mes sea mayor (más reciente) que el penúltimo mes
    """
    if df_resumen.empty:
        return

    # Convertir strings a fechas para comparación
    def mes_a_fecha(mes_str):
        if not mes_str or pd.isna(mes_str):
            return None
        try:
            mes, año = mes_str.split("/")
            return pd.Timestamp(year=int(año), month=int(mes), day=1)
        except:
            return None

    incoherencias = 0
    for idx, row in df_resumen.iterrows():
        ultimo = mes_a_fecha(row.get("Ultimo_Mes_Consumo"))
        penultimo = mes_a_fecha(row.get("Penultimo_Mes_Consumo"))

        if ultimo and penultimo and penultimo > ultimo:
            incoherencias += 1
            logger.warning(
                f"Incoherencia temporal en fila {idx}: Último={row['Ultimo_Mes_Consumo']}, Penúltimo={row['Penultimo_Mes_Consumo']}"
            )

    if incoherencias > 0:
        st.warning(
            f"⚠️ Se encontraron {incoherencias} incoherencias temporales en el resumen"
        )


def calcular_pendiente_por_centro_sin_bloqueo(
    df_todas_sugerencias: pd.DataFrame,
    centros: List[str] = ["1001", "1003", "1004", "1017", "1018", "1022", "1036"],
) -> Dict[str, Dict[str, float]]:
    """
    Calcula la cantidad pendiente por centro sin estatus de bloqueo.
    Retorna un diccionario anidado: {centro: {material_almacen_key: pendiente_total}}
    """
    if df_todas_sugerencias.empty:
        return {}

    try:
        # IMPORTANTE: Filtrar solo las líneas SIN sugerencia (fuente vacía) y SIN bloqueo
        df_sin_bloqueo = df_todas_sugerencias[
            (df_todas_sugerencias[Columnas.FUENTE] == "")  # Solo líneas sin sugerencia
            & (df_todas_sugerencias[Columnas.BLOQUEADO] == "")  # Sin bloqueo
            & (df_todas_sugerencias[Columnas.CANTIDAD_PENDIENTE] > 0)  # Con pendiente
        ].copy()

        if df_sin_bloqueo.empty:
            return {}

        # Crear diccionario para almacenar resultados
        resultados = {centro: {} for centro in centros}

        for centro in centros:
            # Filtrar por centro pedido específico
            df_centro = df_sin_bloqueo[
                (df_sin_bloqueo[Columnas.CENTRO_PEDIDO] == str(centro))
            ]

            if not df_centro.empty:
                # Para evitar duplicados, agrupar por Material, Almacén y PEDIDO primero
                # Esto evita contar múltiples veces el mismo pedido
                df_agrupado = (
                    df_centro.groupby(
                        [
                            Columnas.MATERIAL_SOLICITADO,
                            Columnas.ALMACEN,
                            Columnas.PEDIDO,  # Agrupar también por pedido
                        ]
                    )
                    .agg(
                        {
                            Columnas.CANTIDAD_PENDIENTE: "first"  # Tomar el primer valor (todos son iguales)
                        }
                    )
                    .reset_index()
                )

                # Ahora agrupar solo por Material y Almacén
                df_final = (
                    df_agrupado.groupby(
                        [Columnas.MATERIAL_SOLICITADO, Columnas.ALMACEN]
                    )
                    .agg(
                        Pendiente_Total=(Columnas.CANTIDAD_PENDIENTE, "sum"),
                    )
                    .reset_index()
                )

                # Crear clave única para cada combinación Material/Almacén
                for _, row in df_final.iterrows():
                    material = str(row[Columnas.MATERIAL_SOLICITADO]).strip()
                    almacen = str(row[Columnas.ALMACEN]).strip()
                    clave = f"{material}_{almacen}"

                    # Solo asignar si no existe ya (evitar duplicados)
                    if clave not in resultados[centro]:
                        resultados[centro][clave] = float(row["Pendiente_Total"])
                    else:
                        # Si ya existe, sumar (por si hay múltiples pedidos)
                        resultados[centro][clave] += float(row["Pendiente_Total"])

        return resultados

    except Exception as e:
        logger.error(f"Error al calcular pendiente por centro: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return {}


# =========================
# MODIFICAR: generar_resumen_sin_sugerencias_optimizado para cumplir con los nuevos requisitos
# =========================
def generar_resumen_sin_sugerencias_optimizado(
    df_sugerencias: pd.DataFrame,
    inventario_df: pd.DataFrame,
    df_todas_sugerencias: pd.DataFrame,
    df_facturacion_procesado: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Versión MODIFICADA según los nuevos requisitos:
    1. Debe incluir TODOS los "Material" y "Descripcion" por Centro/Material/Almacén con:
       - Inventario > 0 (ya calculado como: "Libre Utilización" - "Entrega a cliente")
       - O materiales que tengan Pedidos > 0 (sin sugerencia y sin bloqueo)
    """

    # 1. OBTENER MATERIALES CON INVENTARIO > 0
    inventario_materiales = pd.DataFrame()
    if inventario_df is not None and not inventario_df.empty:
        # Filtrar materiales con Libre Utilización > 0 (ya calculado)
        inventario_filtrado = inventario_df[
            inventario_df["Libre Utilización"] > 0
        ].copy()

        if not inventario_filtrado.empty:
            # Crear DataFrame base con materiales de inventario > 0
            inventario_materiales = (
                inventario_filtrado.groupby(["Centro", "Material", "Almacén"])
                .agg(
                    Descripcion=("Descripción", "first"),
                    Libre_Utilizacion_Total=("Libre Utilización", "sum"),
                    Transito_Total=("Cant. en Tránsito", "sum"),
                )
                .reset_index()
            )

            inventario_materiales = inventario_materiales.rename(
                columns={
                    "Centro": "Centro",
                    "Almacén": "Almacen",
                    "Material": "Material",
                    "Descripcion": "Descripcion",
                }
            )
            inventario_materiales["Fuente"] = "Inventario"  # Marcar origen

    # 2. OBTENER MATERIALES CON PEDIDOS > 0 (SIN SUGERENCIA Y SIN BLOQUEO)
    pedidos_materiales = pd.DataFrame()
    if df_sugerencias is not None and not df_sugerencias.empty:
        # Filtrar solo las líneas sin sugerencia (fuente vacía) y SIN BLOQUEO
        df_sin_sugerencia = df_sugerencias[
            (df_sugerencias[Columnas.FUENTE] == "")
            & (df_sugerencias[Columnas.BLOQUEADO] == "")
            & (df_sugerencias[Columnas.CANTIDAD_PENDIENTE] > 0)
        ].copy()

        if not df_sin_sugerencia.empty:
            # Calcular estadísticas de pedidos sin sugerencia
            df_sin_sugerencia["Importe_Calculado"] = (
                df_sin_sugerencia[Columnas.CANTIDAD_PENDIENTE]
                * df_sin_sugerencia[Columnas.PRECIO]
            )

            pedidos_materiales = (
                df_sin_sugerencia.groupby(
                    [
                        Columnas.CENTRO_PEDIDO,
                        Columnas.ALMACEN,
                        Columnas.MATERIAL_SOLICITADO,
                    ]
                )
                .agg(
                    Pedidos=(Columnas.PEDIDO, "nunique"),
                    Descripcion=(Columnas.DESCRIPCION_SOLICITADA, "first"),
                    Cantidad_Pendiente=(Columnas.CANTIDAD_PENDIENTE, "sum"),
                    Importe_Pendiente=("Importe_Calculado", "sum"),
                )
                .reset_index()
            )

            pedidos_materiales = pedidos_materiales.rename(
                columns={
                    Columnas.CENTRO_PEDIDO: "Centro",
                    Columnas.ALMACEN: "Almacen",
                    Columnas.MATERIAL_SOLICITADO: "Material",
                }
            )
            pedidos_materiales["Fuente"] = "Pedidos"  # Marcar origen

    # 3. COMBINAR AMBAS FUENTES (UNIÓN - UNION)
    # Primero, asegurarnos de que ambos DataFrames tengan las mismas columnas
    columnas_comunes = ["Centro", "Almacen", "Material", "Descripcion", "Fuente"]

    # Añadir columnas faltantes a inventario_materiales
    if not inventario_materiales.empty:
        for col in ["Pedidos", "Cantidad_Pendiente", "Importe_Pendiente"]:
            if col not in inventario_materiales.columns:
                inventario_materiales[col] = 0

    # Añadir columnas faltantes a pedidos_materiales
    if not pedidos_materiales.empty:
        for col in ["Libre_Utilizacion_Total", "Transito_Total"]:
            if col not in pedidos_materiales.columns:
                pedidos_materiales[col] = 0

    # 4. COMBINAR AMBOS CONJUNTOS (eliminando duplicados de Centro/Material/Almacen)
    if not inventario_materiales.empty and not pedidos_materiales.empty:
        # Concatenar ambos DataFrames
        combined = pd.concat(
            [inventario_materiales, pedidos_materiales], ignore_index=True
        )

        # Para cada combinación Centro/Material/Almacen, priorizar:
        # 1. Si existe en INVENTARIO, usar esos datos (inventario > 0 tiene prioridad)
        # 2. Si solo existe en PEDIDOS, usar esos datos
        combined = combined.sort_values(
            by=["Centro", "Material", "Almacen", "Fuente"],
            ascending=[
                True,
                True,
                True,
                False,
            ],  # "Inventario" viene antes que "Pedidos" alfabéticamente
        )

        # Eliminar duplicados, manteniendo el primero (prioridad a Inventario)
        grouped = combined.drop_duplicates(
            subset=["Centro", "Material", "Almacen"], keep="first"
        )
    elif not inventario_materiales.empty:
        grouped = inventario_materiales
    elif not pedidos_materiales.empty:
        grouped = pedidos_materiales
    else:
        return pd.DataFrame()

    # 5. RELLENAR VALORES FALTANTES
    # Asegurar que todas las columnas necesarias existan
    for col in [
        "Pedidos",
        "Cantidad_Pendiente",
        "Importe_Pendiente",
        "Libre_Utilizacion_Total",
        "Transito_Total",
    ]:
        if col not in grouped.columns:
            grouped[col] = 0

    # Rellenar descripciones vacías
    if "Descripcion" in grouped.columns:
        grouped["Descripcion"] = grouped["Descripcion"].fillna("")

    # 6. PRECOMPUTAR DATOS DE INVENTARIO PARA CÁLCULOS RÁPIDOS (vectorizado)
    inventario_cache = {}
    descripcion_cache = {}
    transito_cache = {}

    if inventario_df is not None and not inventario_df.empty:
        inv_tmp = inventario_df.copy()
        inv_tmp["_centro"] = inv_tmp["Centro"].astype(str).str.strip()
        inv_tmp["_material"] = inv_tmp["Material"].astype(str).str.strip()
        inv_tmp["_almacen"] = inv_tmp["Almacén"].astype(str).str.strip()

        # inventario_cache: {centro_material_almacen: libre}
        inv_tmp["_key_inv"] = (
            inv_tmp["_centro"] + "_" + inv_tmp["_material"] + "_" + inv_tmp["_almacen"]
        )
        inventario_cache = inv_tmp.set_index("_key_inv")["Libre Utilización"].to_dict()

        # descripcion_cache
        if "Descripción" in inv_tmp.columns:
            desc_sub = inv_tmp[inv_tmp["Descripción"].astype(str).str.strip() != ""]
            if not desc_sub.empty:
                descripcion_cache = (
                    desc_sub.drop_duplicates("_key_inv")
                    .set_index("_key_inv")["Descripción"]
                    .to_dict()
                )

        # transito_cache: {centro_material: {almacen: transito}}
        trans_sub = inv_tmp[inv_tmp["_almacen"].isin(["1030", "1031", "1032"])]
        if not trans_sub.empty:
            trans_grouped = (
                trans_sub.groupby(["_centro", "_material", "_almacen"])[
                    "Cant. en Tránsito"
                ]
                .sum()
                .reset_index()
            )
            for _, row in trans_grouped.iterrows():
                key = f"{row['_centro']}_{row['_material']}"
                if key not in transito_cache:
                    transito_cache[key] = {"1030": 0, "1031": 0, "1032": 0}
                transito_cache[key][row["_almacen"]] = float(row["Cant. en Tránsito"])

    # 7. CALCULAR ESTADÍSTICAS DE CONSUMO (NUEVO)
    estadisticas_consumo_df = None
    if df_facturacion_procesado is not None and not df_facturacion_procesado.empty:
        estadisticas_consumo_df = (
            calcular_estadisticas_consumo_por_centro_material_almacen(
                df_facturacion_procesado
            )
        )

    # 8. AGREGAR ESTADÍSTICAS DE CONSUMO
    if estadisticas_consumo_df is not None and not estadisticas_consumo_df.empty:
        # Hacer merge con estadísticas de consumo
        grouped = pd.merge(
            grouped,
            estadisticas_consumo_df,
            left_on=["Centro", "Material", "Almacen"],
            right_on=["Centro", "Material", "Almacen"],
            how="left",
        )
        # Rellenar valores nulos
        for col in [
            "Promedio_Consumo_12M",
            "Cantidad_Ultimo_Mes",
            "Cantidad_Penultimo_Mes",
        ]:
            if col in grouped.columns:
                grouped[col] = grouped[col].fillna(0)
        for col in ["Ultimo_Mes_Consumo", "Penultimo_Mes_Consumo"]:
            if col in grouped.columns:
                grouped[col] = grouped[col].fillna("")
    else:
        grouped["Promedio_Consumo_12M"] = 0
        grouped["Ultimo_Mes_Consumo"] = ""
        grouped["Penultimo_Mes_Consumo"] = ""
        grouped["Cantidad_Ultimo_Mes"] = 0
        grouped["Cantidad_Penultimo_Mes"] = 0

    # 9. AGREGAR DATOS DE INVENTARIO ESPECÍFICOS POR ALMACÉN (vectorizado)
    def _build_inv_col(col_key_suffix: str, cache: dict) -> pd.Series:
        keys = grouped["Centro"] + "_" + grouped["Material"] + "_" + col_key_suffix
        return keys.map(cache).fillna(0)

    grouped["Inv 1030"] = _build_inv_col("1030", inventario_cache)
    grouped["Inv 1031"] = _build_inv_col("1031", inventario_cache)
    grouped["Inv 1032"] = _build_inv_col("1032", inventario_cache)
    grouped["Inv 1060"] = _build_inv_col("1060", inventario_cache)

    # Tránsito por almacén específico de cada fila
    trans_key = grouped["Centro"] + "_" + grouped["Material"]
    grouped["Cant. en Tránsito"] = trans_key.map(
        lambda k: (
            transito_cache.get(k, {}).get(
                grouped.at[
                    (
                        grouped.index[
                            grouped["Centro"] + "_" + grouped["Material"] == k
                        ].tolist()[0]
                        if (grouped["Centro"] + "_" + grouped["Material"] == k).any()
                        else 0
                    ),
                    "Almacen",
                ],
                0,
            )
            if k in transito_cache
            else 0
        )
    )
    # Vectorized transit per row using Almacen column
    _trans_vals = []
    for _, row in grouped[["Centro", "Material", "Almacen"]].iterrows():
        k = f"{row['Centro']}_{row['Material']}"
        _trans_vals.append(transito_cache.get(k, {}).get(row["Almacen"], 0))
    grouped["Cant. en Tránsito"] = _trans_vals

    # Disponible 1031-1030 y 1031-1032 (fijo por material, independiente del almacén de la fila)
    grouped["Disponible 1031-1030"] = grouped["Material"].map(
        lambda m: inventario_cache.get(f"1031_{m}_1030", 0)
    )
    grouped["Disponible 1031-1032"] = grouped["Material"].map(
        lambda m: inventario_cache.get(f"1031_{m}_1032", 0)
    )

    # 10. CALCULAR MESES DE INVENTARIO
    # Regla: si el almacén del pedido es 1030 → usar Inv 1030
    #        si el almacén del pedido es 1031 → usar Inv 1031
    #        si el almacén del pedido es 1060 → usar Inv 1060
    #        en cualquier otro caso            → usar Inv 1032
    inv_segun_almacen = np.select(
        [
            grouped["Almacen"].astype(str).str.strip() == "1030",
            grouped["Almacen"].astype(str).str.strip() == "1031",
            grouped["Almacen"].astype(str).str.strip() == "1060",
        ],
        [
            grouped["Inv 1030"],
            grouped["Inv 1031"],
            grouped["Inv 1060"],
        ],
        default=grouped["Inv 1032"],
    )
    consumo_prom = grouped["Promedio_Consumo_12M"]
    grouped["Meses_Inventario"] = np.where(
        consumo_prom > 0,
        (inv_segun_almacen / consumo_prom).round(2),
        np.where(inv_segun_almacen == 0, 0.0, 999.0),
    )

    # 11. CALCULAR PENDIENTE POR CENTRO SIN BLOQUEO - VERSIÓN CORREGIDA
    pendiente_por_centro_dict = None
    if df_todas_sugerencias is not None and not df_todas_sugerencias.empty:
        pendiente_por_centro_dict = calcular_pendiente_por_centro_sin_bloqueo(
            df_todas_sugerencias
        )

    # 12. AGREGAR PENDIENTE POR CENTRO - versión vectorizada
    centros_interes = ["1001", "1003", "1004", "1017", "1018", "1022", "1036"]

    if pendiente_por_centro_dict:
        for centro in centros_interes:
            col_name = f"Pendiente {centro}"
            centro_dict = pendiente_por_centro_dict.get(centro, {})
            if centro_dict:
                # Crear máscara: solo filas donde el Centro de la fila == centro
                mask_centro = grouped["Centro"].astype(str).str.strip() == str(centro)
                claves = (
                    grouped["Material"].astype(str).str.strip()
                    + "_"
                    + grouped["Almacen"].astype(str).str.strip()
                )
                valores = claves.map(centro_dict).fillna(0)
                grouped[col_name] = np.where(mask_centro, valores, 0)
            else:
                grouped[col_name] = 0
    else:
        for centro in centros_interes:
            grouped[f"Pendiente {centro}"] = 0

    # 13. ORDENAR COLUMNAS SEGÚN LO SOLICITADO
    columnas_orden = [
        "Centro",
        "Almacen",
        "Pedidos",  # Mantener para referencia
        "Material",
        "Descripcion",
        "Cantidad_Pendiente",
        "Importe_Pendiente",
        "Promedio_Consumo_12M",
        "Ultimo_Mes_Consumo",
        "Cantidad_Ultimo_Mes",
        "Penultimo_Mes_Consumo",
        "Cantidad_Penultimo_Mes",
        "Meses_Inventario",
        "Inv 1030",
        "Inv 1031",
        "Inv 1032",
        "Inv 1060",
        "Cant. en Tránsito",
        "Disponible 1031-1030",
        "Disponible 1031-1032",
    ]

    # Agregar columnas de pendiente por centro
    for centro in centros_interes:
        columnas_orden.append(f"Pendiente {centro}")

    # Agregar columna de fuente para depuración (opcional)
    columnas_orden.append("Fuente")

    # Asegurar que todas las columnas existan
    for col in columnas_orden:
        if col not in grouped.columns:
            if col in [
                "Descripcion",
                "Centro",
                "Almacen",
                "Material",
                "Ultimo_Mes_Consumo",
                "Penultimo_Mes_Consumo",
                "Fuente",
            ]:
                grouped[col] = ""
            elif col == "Meses_Inventario":
                grouped[col] = 0.0
            else:
                grouped[col] = 0

    # 14. ORDENAR POR CENTRO, ALMACEN, MATERIAL
    grouped = grouped.sort_values(
        by=["Centro", "Almacen", "Material"], ascending=[True, True, True]
    )

    return grouped[columnas_orden]


# =========================
# MODIFICAR: Función exportar_a_excel para incluir la hoja de resumen modificada
# =========================
def exportar_a_excel(
    df_todas_sugerencias: pd.DataFrame = None,
    df_resumen_sin_sugerencias: pd.DataFrame = None,
    df_reporte_consumo: pd.DataFrame = None,
    df_sug_consumo: pd.DataFrame = None,
) -> bytes:
    """Exporta los reportes seleccionados a Excel"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Agregar hoja "Todas las Sugerencias" si se proporciona
        if df_todas_sugerencias is not None and not df_todas_sugerencias.empty:
            df_todas_sugerencias.to_excel(
                writer, sheet_name="Todas las Sugerencias", index=False
            )

        # Agregar hoja "Resumen Sin Sugerencias" si se proporciona (CON LOS CAMBIOS)
        if (
            df_resumen_sin_sugerencias is not None
            and not df_resumen_sin_sugerencias.empty
        ):
            df_resumen_sin_sugerencias.to_excel(
                writer, sheet_name="Resumen Sin Sugerencias", index=False
            )

        # Agregar hoja "Reporte de Consumo" si se proporciona
        if df_reporte_consumo is not None and not df_reporte_consumo.empty:
            df_reporte_consumo.to_excel(
                writer, sheet_name="Reporte de Consumo", index=False
            )

        # Agregar hoja "Sug Reporte Consumo" si se proporciona (nueva)
        if df_sug_consumo is not None and not df_sug_consumo.empty:
            df_sug_consumo.to_excel(
                writer, sheet_name="Sug Reporte Consumo", index=False
            )

    return output.getvalue()


def exportar_reporte_individual(df_reporte: pd.DataFrame, nombre_reporte: str) -> bytes:
    """Exporta un solo reporte a Excel"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_reporte.to_excel(
            writer,
            sheet_name=nombre_reporte[:31],  # Excel limita a 31 caracteres
            index=False,
        )

    return output.getvalue()


# ------------------------------------------------------------------------------
# Interfaz de Streamlit
# ------------------------------------------------------------------------------

# Sidebar con configuración
st.sidebar.header("Configuración")

# Selección de fuentes activas
fuentes_disponibles = [
    "Corta caducidad",
    "Lento mov",
    "Cosmopark",
    "Sustituto",
    "PNC",
    "Caduco",
]
fuentes_activas = st.sidebar.multiselect(
    "Fuentes a considerar:", options=fuentes_disponibles, default=fuentes_disponibles
)

# NUEVO: Selección de reportes a generar
st.sidebar.header("Reportes a Generar")
generar_todas_sugerencias_report = st.sidebar.checkbox(
    "Generar 'Todas las Sugerencias'", value=True
)
generar_resumen_sin_sugerencias_report = st.sidebar.checkbox(
    "Generar 'Resumen Sin Sugerencias'", value=True
)

generar_reporte_consumo_report = st.sidebar.checkbox(
    "Generar 'Reporte de Consumo'", value=False
)

generar_sug_consumo_report = st.sidebar.checkbox(
    "Generar 'Sugerencias desde Consumo'",
    value=False,
    help=(
        "Requiere tener activo 'Generar Reporte de Consumo' y cargar el archivo de facturación. "
        "Aplica la misma lógica de sugerencias del reporte principal, pero sobre los registros "
        "del Reporte de Consumo."
    ),
)

# Modo depuración para ver columnas
modo_depuracion = st.sidebar.checkbox("Modo depuración (ver columnas)", value=False)

# Carga de archivos
# ------------------------------------------------------------------------------
# MODIFICADO: Carga de 3 archivos separados
# ------------------------------------------------------------------------------
st.header("Carga de Archivos Separados")

# 1. Archivo con Seg pedidos
archivo_principal = st.file_uploader(
    "1. Archivo con hoja 'Seg pedidos' o 'sheets1' (Excel)",
    type=["xlsx", "xls"],
    key="principal",
)

# 2. Archivo con Inventario
archivo_inventario = st.file_uploader(
    "2. Archivo con pestaña 'Inventario' o 'sheets1' (Excel)",
    type=["xlsx", "xls"],
    key="inventario",
)

# 3. Archivo con hojas externas
archivo_externas = st.file_uploader(
    "3. Archivo con pestañas externas (Corta caducidad, Lento mov, etc.) (Excel)",
    type=["xlsx", "xls"],
    key="externas",
)

_necesita_facturacion = generar_reporte_consumo_report or generar_sug_consumo_report
if _necesita_facturacion:
    archivo_facturacion = st.file_uploader(
        "4. Archivo con pestaña 'Facturacion' o 'sheets1' (Excel)",
        type=["xlsx", "xls"],
        key="facturacion",
    )
else:
    archivo_facturacion = None
    st.info(
        "Para cargar el archivo de facturación, active 'Generar Reporte de Consumo' "
        "o 'Sugerencias desde Consumo' en la barra lateral"
    )

# Verificar que se hayan subido los 3 archivos obligatorios
# (facturación solo es obligatoria si se activa un reporte que la necesita)
if (
    archivo_principal
    and archivo_inventario
    and archivo_externas
    and (
        not _necesita_facturacion
        or (_necesita_facturacion and archivo_facturacion)
    )
):
    timer_total = Timer()

    # ── Inicializar cache ────────────────────────────────────────────────────
    if "cache_inicializado" not in st.session_state:
        st.session_state.cache_inicializado = True
        st.session_state.cache_pedidos = None
        st.session_state.cache_inventario = None
        st.session_state.cache_externas = None
        st.session_state.cache_facturacion = None

    if "reportes_generados" not in st.session_state:
        st.session_state.reportes_generados = {}

    usar_cache = st.checkbox(
        "Usar cache de datos procesados (acelera reprocesamiento)", value=True
    )

    try:
        # ════════════════════════════════════════════════════════════════════
        # FASE 1 – CARGA Y PROCESAMIENTO DE ARCHIVOS (en paralelo)
        # ════════════════════════════════════════════════════════════════════
        with st.status(
            "📂 Cargando y procesando archivos…", expanded=True
        ) as status_carga:

            # ── Función para cargar pedidos ──────────────────────────────
            def _cargar_pedidos():
                t = Timer()
                xls = pd.ExcelFile(archivo_principal)
                sheet_map = {s.strip().casefold(): s for s in xls.sheet_names}
                hoja = None
                for candidato in ["seg pedidos", "sheets1"]:
                    if candidato in sheet_map:
                        hoja = sheet_map[candidato]
                        break
                if hoja is None:
                    cols_min = {"Pedido", "Material", "Centro"}
                    for sh in xls.sheet_names:
                        try:
                            cols = set(pd.read_excel(xls, sh, nrows=0).columns)
                            if cols_min.issubset(cols):
                                hoja = sh
                                break
                        except Exception:
                            pass
                if hoja is None:
                    raise ValueError(
                        f"No se encontró hoja de pedidos. Hojas: {xls.sheet_names}"
                    )
                df = pd.read_excel(xls, hoja)
                df.columns = [
                    col.replace("Almacen", "Almacén").replace("Almaçen", "Almacén")
                    for col in df.columns
                ]
                col_gpo = encontrar_columna_por_patron(
                    df, ["gpo.vdor", "gpo. vdor", "gpo vdor", "grupo vendedor", "vdor"]
                )
                if "Gpo.Vdor." not in df.columns:
                    df["Gpo.Vdor."] = df[col_gpo] if col_gpo else ""
                df["Gpo.Vdor."] = (
                    df["Gpo.Vdor."]
                    .astype(str)
                    .str.strip()
                    .replace({"nan": "", "None": ""})
                )
                for col in ["Centro", "Material", "Almacén"]:
                    if col in df.columns:
                        df[col] = normalizar_ids(df[col])
                return df, hoja, t.elapsed()

            # ── Función para cargar inventario ───────────────────────────
            def _cargar_inventario():
                t = Timer()
                xls = pd.ExcelFile(archivo_inventario)
                hoja = None
                for h in xls.sheet_names:
                    if "inventario" in h.lower() or "sheets1" in h.lower():
                        hoja = h
                        break
                if hoja is None:
                    hoja = xls.sheet_names[0]
                df_raw = pd.read_excel(xls, hoja)
                df = procesar_hoja_inventario_ajustada(df_raw)
                return df, hoja, t.elapsed()

            # ── Función para cargar hojas externas ───────────────────────
            def _cargar_externas():
                t = Timer()
                xls = pd.ExcelFile(archivo_externas)
                hojas = {}
                for hoja in xls.sheet_names:
                    if "inventario" not in hoja.lower() and hoja in fuentes_disponibles:
                        df_hoja = pd.read_excel(xls, hoja)
                        if generar_todas_sugerencias_report or generar_sug_consumo_report:
                            hojas[hoja] = procesar_hoja_externa(df_hoja, hoja)
                return hojas, t.elapsed()

            # ── Función para cargar facturación ──────────────────────────
            def _cargar_facturacion():
                if not (generar_reporte_consumo_report and archivo_facturacion):
                    return None, None, "–"
                t = Timer()
                xls = pd.ExcelFile(archivo_facturacion)
                hoja = None
                for h in xls.sheet_names:
                    if "facturacion" in h.lower() or "sheets1" in h.lower():
                        hoja = h
                        break
                if hoja is None:
                    hoja = xls.sheet_names[0]
                df_raw = pd.read_excel(xls, hoja)
                df = procesar_datos_facturacion(df_raw)
                return df, hoja, t.elapsed()

            # ── Usar cache o cargar en paralelo ──────────────────────────
            if usar_cache and st.session_state.cache_pedidos is not None:
                pedidos_df = st.session_state.cache_pedidos
                inventario_df = st.session_state.cache_inventario
                hojas_externas = st.session_state.cache_externas
                df_facturacion_procesado = st.session_state.cache_facturacion

                st.write("✅ Datos cargados desde cache")
                col1, col2, col3 = st.columns(3)
                col1.metric("Pedidos (cache)", len(pedidos_df))
                col2.metric("Inventario (cache)", len(inventario_df))
                col3.metric("Hojas externas (cache)", len(hojas_externas))
                hoja_pedidos_nombre = "cache"
            else:
                st.write("⚙️ Cargando archivos en paralelo…")
                carga_progress = st.progress(0.0)

                resultados_carga = {}
                errores_carga = {}

                tareas = {
                    "pedidos": _cargar_pedidos,
                    "inventario": _cargar_inventario,
                    "externas": _cargar_externas,
                    "facturacion": _cargar_facturacion,
                }

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(fn): nombre for nombre, fn in tareas.items()
                    }
                    completados = 0
                    for future in as_completed(futures):
                        nombre = futures[future]
                        completados += 1
                        carga_progress.progress(completados / len(tareas))
                        try:
                            resultados_carga[nombre] = future.result()
                        except Exception as e:
                            errores_carga[nombre] = str(e)

                if "pedidos" in errores_carga:
                    st.error(f"Error cargando pedidos: {errores_carga['pedidos']}")
                    st.stop()

                pedidos_df, hoja_pedidos_nombre, t_ped = resultados_carga["pedidos"]
                inventario_df, _, t_inv = resultados_carga["inventario"]
                hojas_externas, t_ext = resultados_carga["externas"]
                df_facturacion_procesado, _, t_fac = resultados_carga.get(
                    "facturacion", (None, None, "–")
                )

                carga_progress.empty()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pedidos", f"{len(pedidos_df):,}", delta=f"⏱ {t_ped}")
                c2.metric("Inventario", f"{len(inventario_df):,}", delta=f"⏱ {t_inv}")
                c3.metric("Hojas externas", len(hojas_externas), delta=f"⏱ {t_ext}")
                c4.metric(
                    "Facturación",
                    (
                        f"{len(df_facturacion_procesado):,}"
                        if df_facturacion_procesado is not None
                        else "–"
                    ),
                    delta=f"⏱ {t_fac}",
                )

                # Guardar en cache
                st.session_state.cache_pedidos = pedidos_df
                st.session_state.cache_inventario = inventario_df
                st.session_state.cache_externas = hojas_externas
                if df_facturacion_procesado is not None:
                    st.session_state.cache_facturacion = df_facturacion_procesado

            if modo_depuracion:
                with st.expander("🔍 Columnas detectadas (debug)"):
                    st.write("**Pedidos:**", pedidos_df.columns.tolist())
                    st.write("**Inventario:**", inventario_df.columns.tolist())
                    for h, df_h in hojas_externas.items():
                        st.write(f"**{h}:**", df_h.columns.tolist())

            status_carga.update(
                label=f"✅ Archivos listos ({timer_total.elapsed()})", state="complete"
            )

        # ════════════════════════════════════════════════════════════════════
        # FASE 2 – REPORTE DE CONSUMO
        # ════════════════════════════════════════════════════════════════════
        df_reporte_consumo = None
        if (
            generar_reporte_consumo_report
            and df_facturacion_procesado is not None
            and not df_facturacion_procesado.empty
        ):
            with st.status(
                "📈 Generando Reporte de Consumo…", expanded=False
            ) as status_consumo:
                t2 = Timer()
                try:
                    df_reporte_consumo = generar_reporte_consumo(
                        df_facturacion_procesado
                    )
                    if df_reporte_consumo is not None and not df_reporte_consumo.empty:
                        status_consumo.update(
                            label=f"✅ Reporte de Consumo listo — {len(df_reporte_consumo):,} materiales ({t2.elapsed()})",
                            state="complete",
                        )
                    else:
                        status_consumo.update(
                            label="⚠️ Reporte de consumo vacío", state="error"
                        )
                except Exception as e:
                    st.error(f"Error en reporte de consumo: {e}")
                    logger.error(f"Error reporte consumo: {e}", exc_info=True)
                    status_consumo.update(
                        label="❌ Error en Reporte de Consumo", state="error"
                    )
        elif generar_reporte_consumo_report and archivo_facturacion is None:
            st.warning(
                "Para generar el reporte de consumo, cargue el archivo de facturación."
            )

        # ════════════════════════════════════════════════════════════════════
        # FASE 3 – TODAS LAS SUGERENCIAS
        # ════════════════════════════════════════════════════════════════════
        df_todas_sugerencias = None
        if generar_todas_sugerencias_report:
            with st.status(
                "💡 Generando Todas las Sugerencias…", expanded=True
            ) as status_sug:
                t3 = Timer()
                try:
                    df_todas_sugerencias = generar_todas_sugerencias(
                        pedidos_df, hojas_externas, fuentes_activas, inventario_df
                    )
                    if (
                        df_todas_sugerencias is not None
                        and not df_todas_sugerencias.empty
                    ):
                        n_con = (df_todas_sugerencias[Columnas.FUENTE] != "").sum()
                        n_sin = (df_todas_sugerencias[Columnas.FUENTE] == "").sum()
                        n_bloq = (df_todas_sugerencias[Columnas.BLOQUEADO] != "").sum()
                        status_sug.update(
                            label=f"✅ Sugerencias listas — {len(df_todas_sugerencias):,} líneas ({t3.elapsed()})",
                            state="complete",
                        )
                    else:
                        status_sug.update(
                            label="⚠️ Sin sugerencias generadas", state="error"
                        )
                except Exception as e:
                    st.error(f"Error al generar sugerencias: {e}")
                    logger.error(f"Error sugerencias: {e}", exc_info=True)
                    status_sug.update(label="❌ Error en Sugerencias", state="error")

        # ════════════════════════════════════════════════════════════════════
        # FASE 4 – RESUMEN SIN SUGERENCIAS
        # ════════════════════════════════════════════════════════════════════
        df_resumen_sin_sugerencias = None
        if (
            generar_resumen_sin_sugerencias_report
            and df_todas_sugerencias is not None
            and not df_todas_sugerencias.empty
        ):
            with st.status(
                "📋 Generando Resumen Sin Sugerencias…", expanded=False
            ) as status_res:
                t4 = Timer()
                try:
                    facturacion_para_resumen = (
                        df_facturacion_procesado
                        if df_facturacion_procesado is not None
                        else None
                    )
                    df_resumen_sin_sugerencias = (
                        generar_resumen_sin_sugerencias_optimizado(
                            df_todas_sugerencias,
                            inventario_df,
                            df_todas_sugerencias,
                            facturacion_para_resumen,
                        )
                    )
                    if (
                        df_resumen_sin_sugerencias is not None
                        and not df_resumen_sin_sugerencias.empty
                    ):
                        status_res.update(
                            label=f"✅ Resumen listo — {len(df_resumen_sin_sugerencias):,} registros ({t4.elapsed()})",
                            state="complete",
                        )
                    else:
                        status_res.update(label="⚠️ Resumen vacío", state="error")
                except Exception as e:
                    st.error(f"Error al generar resumen: {e}")
                    logger.error(f"Error resumen: {e}", exc_info=True)
                    status_res.update(label="❌ Error en Resumen", state="error")

        # ════════════════════════════════════════════════════════════════════
        # POST-PROCESO: Enriquecer Todas las Sugerencias con consumo del Resumen
        # ════════════════════════════════════════════════════════════════════
        if (
            df_todas_sugerencias is not None
            and not df_todas_sugerencias.empty
            and (
                (df_resumen_sin_sugerencias is not None and not df_resumen_sin_sugerencias.empty)
                or (df_reporte_consumo is not None and not df_reporte_consumo.empty)
            )
        ):
            with st.status(
                "🔗 Enriqueciendo Sugerencias con datos de consumo…", expanded=False
            ) as status_enr:
                t_enr = Timer()
                try:
                    df_todas_sugerencias = enriquecer_sugerencias_con_consumo(
                        df_todas_sugerencias,
                        df_resumen_sin_sugerencias,
                        df_facturacion_procesado if df_facturacion_procesado is not None else None,
                        df_reporte_consumo if df_reporte_consumo is not None else None,
                    )
                    status_enr.update(
                        label=f"✅ Meses_Inventario calculado por almacén ({t_enr.elapsed()})",
                        state="complete",
                    )
                except Exception as e:
                    logger.warning(f"Post-proceso consumo falló: {e}")
                    status_enr.update(label="⚠️ Enriquecimiento omitido", state="error")

        # ════════════════════════════════════════════════════════════════════
        # FASE 5 – SUGERENCIAS DESDE REPORTE DE CONSUMO (nuevo reporte)
        # ════════════════════════════════════════════════════════════════════
        df_sug_consumo = None
        if (
            generar_sug_consumo_report
            and df_reporte_consumo is not None
            and not df_reporte_consumo.empty
        ):
            with st.status(
                "🔎 Generando Sugerencias desde Reporte de Consumo…", expanded=True
            ) as status_sug_rc:
                t5 = Timer()
                try:
                    df_sug_consumo = generar_sugerencias_desde_reporte_consumo(
                        df_reporte_consumo=df_reporte_consumo,
                        hojas_externas=hojas_externas,
                        fuentes_activas=fuentes_activas,
                        inventario_df=inventario_df,
                        df_resumen=df_resumen_sin_sugerencias,
                    )
                    if df_sug_consumo is not None and not df_sug_consumo.empty:
                        n_con_rc = int(
                            (df_sug_consumo[Columnas.FUENTE] != "").sum()
                        )
                        n_sin_rc = int(
                            (df_sug_consumo[Columnas.FUENTE] == "").sum()
                        )
                        status_sug_rc.update(
                            label=(
                                f"✅ Sugerencias desde Consumo listas — "
                                f"{len(df_sug_consumo):,} líneas "
                                f"({n_con_rc:,} con sugerencia, "
                                f"{n_sin_rc:,} sin cobertura) "
                                f"({t5.elapsed()})"
                            ),
                            state="complete",
                        )
                    else:
                        status_sug_rc.update(
                            label="⚠️ Sin sugerencias generadas desde Consumo",
                            state="error",
                        )
                except Exception as e:
                    st.error(f"Error al generar Sugerencias desde Consumo: {e}")
                    logger.error(
                        f"Error Sugerencias desde Consumo: {e}", exc_info=True
                    )
                    status_sug_rc.update(
                        label="❌ Error en Sugerencias desde Consumo", state="error"
                    )
        elif generar_sug_consumo_report and (
            df_reporte_consumo is None or df_reporte_consumo.empty
        ):
            st.warning(
                "Para generar 'Sugerencias desde Consumo', primero activa y genera "
                "'Reporte de Consumo' con el archivo de facturación cargado."
            )

        # ════════════════════════════════════════════════════════════════════
        # RESULTADOS Y DESCARGAS
        # ════════════════════════════════════════════════════════════════════
        st.header(f"📊 Reportes Generados  ⏱ Total: {timer_total.elapsed()}")
        st.session_state.reportes_generados = {}

        # ── Reporte de Consumo ───────────────────────────────────────────
        if (
            generar_reporte_consumo_report
            and df_reporte_consumo is not None
            and not df_reporte_consumo.empty
        ):
            st.session_state.reportes_generados["consumo"] = df_reporte_consumo
            with st.expander("✅ Reporte de Consumo", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Materiales únicos", df_reporte_consumo["Material"].nunique()
                )
                col2.metric(
                    "Destinatarios", df_reporte_consumo["Destinatario"].nunique()
                )
                col3.metric(
                    "Consumo total mensual",
                    f"{df_reporte_consumo['Consumo_promedio_mensual'].sum():,.0f}",
                )
                st.dataframe(df_reporte_consumo.head(10), use_container_width=True)
                excel_bytes_consumo = exportar_reporte_individual(
                    df_reporte_consumo, "Reporte de Consumo"
                )
                st.download_button(
                    "📥 Descargar Reporte de Consumo",
                    data=excel_bytes_consumo,
                    file_name="Reporte_Consumo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_consumo",
                )

        # ── Todas las Sugerencias (4 pestañas) ───────────────────────────
        if (
            generar_todas_sugerencias_report
            and df_todas_sugerencias is not None
            and not df_todas_sugerencias.empty
        ):
            st.session_state.reportes_generados["sugerencias"] = df_todas_sugerencias
            with st.expander("✅ Todas las Sugerencias", expanded=True):
                # ── KPIs superiores ──────────────────────────────────────
                n_con = int((df_todas_sugerencias[Columnas.FUENTE] != "").sum())
                n_sin = int((df_todas_sugerencias[Columnas.FUENTE] == "").sum())
                n_bloq = int((df_todas_sugerencias[Columnas.BLOQUEADO] != "").sum())
                n_ped = int(df_todas_sugerencias[Columnas.PEDIDO].nunique())
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pedidos únicos", f"{n_ped:,}")
                c2.metric("Con sugerencia", f"{n_con:,}")
                c3.metric("Sin cobertura", f"{n_sin:,}")
                c4.metric("Con bloqueo", f"{n_bloq:,}")

                # ── 4 Pestañas ───────────────────────────────────────────
                tab_sug, tab_sin, tab_stock, tab_cli = st.tabs(
                    [
                        "📦 Sugerencias activas",
                        "⚠️ Sin cobertura",
                        "📈 Consumo vs stock",
                        "👤 Vista por cliente",
                    ]
                )

                # ── TAB 1: Sugerencias activas ───────────────────────────
                with tab_sug:
                    st.caption("Pedidos con al menos una fuente de oferta disponible")

                    # Filtros
                    col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
                    with col_f1:
                        fuentes_disp = sorted(
                            df_todas_sugerencias[Columnas.FUENTE]
                            .replace("", "Sin sugerencia")
                            .unique()
                        )
                        fuente_sel = st.selectbox(
                            "Fuente", ["Todas"] + list(fuentes_disp), key="tab1_fuente"
                        )
                    with col_f2:
                        centros_disp_sel = sorted(
                            df_todas_sugerencias[Columnas.CENTRO_PEDIDO].unique()
                        )
                        centro_sel = st.selectbox(
                            "Centro",
                            ["Todos"] + list(centros_disp_sel),
                            key="tab1_centro",
                        )
                    with col_f3:
                        busqueda = st.text_input(
                            "Buscar material o descripción", key="tab1_buscar"
                        )

                    df_tab1 = df_todas_sugerencias[
                        df_todas_sugerencias[Columnas.FUENTE] != ""
                    ].copy()
                    if fuente_sel != "Todas":
                        df_tab1 = df_tab1[df_tab1[Columnas.FUENTE] == fuente_sel]
                    if centro_sel != "Todos":
                        df_tab1 = df_tab1[df_tab1[Columnas.CENTRO_PEDIDO] == centro_sel]
                    if busqueda:
                        mask_b = (
                            df_tab1[Columnas.MATERIAL_SOLICITADO]
                            .astype(str)
                            .str.contains(busqueda, case=False, na=False)
                        ) | (
                            df_tab1[Columnas.DESCRIPCION_SOLICITADA]
                            .astype(str)
                            .str.contains(busqueda, case=False, na=False)
                        )
                        df_tab1 = df_tab1[mask_b]

                    # Columnas clave para el vendedor (orden de trabajo)
                    cols_vista = [
                        Columnas.PEDIDO,
                        Columnas.DESTINATARIO,
                        Columnas.MATERIAL_SOLICITADO,
                        Columnas.DESCRIPCION_SOLICITADA,
                        Columnas.CENTRO_PEDIDO,
                        Columnas.ALMACEN,
                        Columnas.CANTIDAD_PENDIENTE,
                        Columnas.FUENTE,
                        Columnas.CANTIDAD_OFERTAR,
                        Columnas.DISPONIBLE,
                        Columnas.LOTE,
                        Columnas.FECHA_CADUCIDAD,
                        Columnas.INV_1030,
                        Columnas.INV_1031,
                        Columnas.INV_1060,
                        Columnas.CONSUMO_DESTINATARIO_12M,
                        Columnas.PROMEDIO_CONSUMO_12M,
                        Columnas.MESES_INVENTARIO,
                    ]
                    cols_vista = [c for c in cols_vista if c in df_tab1.columns]
                    st.dataframe(
                        df_tab1[cols_vista].reset_index(drop=True),
                        use_container_width=True,
                        height=420,
                    )
                    st.caption(f"{len(df_tab1):,} líneas mostradas")

                # ── TAB 2: Sin cobertura ─────────────────────────────────
                with tab_sin:
                    st.caption(
                        "Pedidos sin ninguna fuente disponible — muestra brecha y materiales afectados"
                    )
                    df_sin_cob = df_todas_sugerencias[
                        df_todas_sugerencias[Columnas.FUENTE] == ""
                    ].copy()

                    if df_sin_cob.empty:
                        st.success("¡Todos los pedidos tienen al menos una sugerencia!")
                    else:
                        # Agrupar por material para mostrar brecha total
                        df_brecha = (
                            df_sin_cob.groupby(
                                [
                                    Columnas.MATERIAL_SOLICITADO,
                                    Columnas.DESCRIPCION_SOLICITADA,
                                    Columnas.CENTRO_PEDIDO,
                                    Columnas.ALMACEN,
                                ]
                            )
                            .agg(
                                Clientes_afectados=(Columnas.DESTINATARIO, "nunique"),
                                Pedidos_afectados=(Columnas.PEDIDO, "nunique"),
                                Pendiente_total=(Columnas.CANTIDAD_PENDIENTE, "sum"),
                                Inv_1030=(Columnas.INV_1030, "first"),
                                Inv_1031=(Columnas.INV_1031, "first"),
                                Inv_1060=(Columnas.INV_1060, "first"),
                                Consumo_12M=(Columnas.PROMEDIO_CONSUMO_12M, "first"),
                                Meses_Inv=(Columnas.MESES_INVENTARIO, "first"),
                            )
                            .reset_index()
                        )

                        # Calcular brecha = pendiente - inventario disponible según almacén
                        def _inv_almacen(row):
                            alm = str(row[Columnas.ALMACEN]).strip()
                            if alm == "1030":
                                return row["Inv_1030"]
                            elif alm == "1031":
                                return row["Inv_1031"]
                            elif alm == "1060":
                                return row["Inv_1060"]
                            return 0

                        df_brecha["Inv_disponible"] = df_brecha.apply(
                            _inv_almacen, axis=1
                        )
                        df_brecha["Brecha"] = (
                            df_brecha["Pendiente_total"] - df_brecha["Inv_disponible"]
                        )
                        df_brecha = df_brecha.sort_values("Brecha", ascending=False)

                        kb1, kb2, kb3 = st.columns(3)
                        kb1.metric(
                            "Materiales sin cobertura",
                            df_brecha[Columnas.MATERIAL_SOLICITADO].nunique(),
                        )
                        kb2.metric(
                            "Clientes afectados",
                            int(df_brecha["Clientes_afectados"].sum()),
                        )
                        kb3.metric(
                            "Pendiente total sin cubrir",
                            f"{df_brecha['Pendiente_total'].sum():,.0f}",
                        )

                        st.dataframe(
                            df_brecha.rename(
                                columns={
                                    Columnas.MATERIAL_SOLICITADO: "Material",
                                    Columnas.DESCRIPCION_SOLICITADA: "Descripción",
                                    Columnas.CENTRO_PEDIDO: "Centro",
                                    Columnas.ALMACEN: "Almacén",
                                }
                            ).reset_index(drop=True),
                            use_container_width=True,
                            height=420,
                        )

                # ── TAB 3: Consumo vs stock ──────────────────────────────
                with tab_stock:
                    st.caption(
                        "Muestra el Promedio_Consumo_12M (por Centro/Material/Almacén) "
                        "frente al inventario del almacén correspondiente al pedido"
                    )
                    if (
                        df_resumen_sin_sugerencias is not None
                        and not df_resumen_sin_sugerencias.empty
                    ):
                        cols_stock = [
                            c
                            for c in [
                                "Centro",
                                "Almacen",
                                "Material",
                                "Descripcion",
                                "Promedio_Consumo_12M",
                                "Ultimo_Mes_Consumo",
                                "Cantidad_Ultimo_Mes",
                                "Penultimo_Mes_Consumo",
                                "Cantidad_Penultimo_Mes",
                                "Meses_Inventario",
                                "Inv 1030",
                                "Inv 1031",
                                "Inv 1032",
                                "Inv 1060",
                            ]
                            if c in df_resumen_sin_sugerencias.columns
                        ]
                        # Filtro de alertas
                        alerta_sel = st.selectbox(
                            "Filtrar por alerta de inventario",
                            [
                                "Todos",
                                "Crítico (< 1 mes)",
                                "Bajo (1–3 meses)",
                                "OK (> 3 meses)",
                            ],
                            key="tab3_alerta",
                        )
                        df_stock = df_resumen_sin_sugerencias[cols_stock].copy()
                        if alerta_sel == "Crítico (< 1 mes)":
                            df_stock = df_stock[df_stock["Meses_Inventario"] < 1]
                        elif alerta_sel == "Bajo (1–3 meses)":
                            df_stock = df_stock[
                                (df_stock["Meses_Inventario"] >= 1)
                                & (df_stock["Meses_Inventario"] <= 3)
                            ]
                        elif alerta_sel == "OK (> 3 meses)":
                            df_stock = df_stock[df_stock["Meses_Inventario"] > 3]

                        k1, k2, k3 = st.columns(3)
                        k1.metric(
                            "Críticos (< 1 mes)",
                            int(
                                (
                                    df_resumen_sin_sugerencias.get(
                                        "Meses_Inventario", pd.Series()
                                    )
                                    < 1
                                ).sum()
                            ),
                        )
                        k2.metric(
                            "Bajos (1–3 meses)",
                            int(
                                (
                                    (
                                        df_resumen_sin_sugerencias.get(
                                            "Meses_Inventario", pd.Series()
                                        )
                                        >= 1
                                    )
                                    & (
                                        df_resumen_sin_sugerencias.get(
                                            "Meses_Inventario", pd.Series()
                                        )
                                        <= 3
                                    )
                                ).sum()
                            ),
                        )
                        k3.metric(
                            "OK (> 3 meses)",
                            int(
                                (
                                    df_resumen_sin_sugerencias.get(
                                        "Meses_Inventario", pd.Series()
                                    )
                                    > 3
                                ).sum()
                            ),
                        )

                        st.dataframe(
                            df_stock.sort_values("Meses_Inventario").reset_index(
                                drop=True
                            ),
                            use_container_width=True,
                            height=420,
                        )
                    else:
                        st.info(
                            "Genera también el Resumen Sin Sugerencias para ver esta vista."
                        )

                # ── TAB 4: Vista por cliente ─────────────────────────────
                with tab_cli:
                    st.caption(
                        "Filtra todos los pedidos abiertos de un cliente para ver su perfil de demanda cruzado con el inventario disponible"
                    )
                    clientes_lista = sorted(
                        df_todas_sugerencias[Columnas.DESTINATARIO]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .replace("", pd.NA)
                        .dropna()
                        .unique()
                    )
                    cliente_elegido = st.selectbox(
                        "Selecciona un destinatario",
                        ["— Elige un cliente —"] + list(clientes_lista),
                        key="tab4_cliente",
                    )
                    if cliente_elegido != "— Elige un cliente —":
                        df_cli = df_todas_sugerencias[
                            df_todas_sugerencias[Columnas.DESTINATARIO]
                            .astype(str)
                            .str.strip()
                            == cliente_elegido
                        ].copy()

                        razon = (
                            df_cli[Columnas.RAZON_SOCIAL].iloc[0]
                            if Columnas.RAZON_SOCIAL in df_cli.columns
                            and len(df_cli) > 0
                            else ""
                        )
                        n_ped_cli = df_cli[Columnas.PEDIDO].nunique()
                        pend_total = df_cli[Columnas.CANTIDAD_PENDIENTE].sum()
                        n_con_cli = (df_cli[Columnas.FUENTE] != "").sum()
                        n_sin_cli = (df_cli[Columnas.FUENTE] == "").sum()

                        st.markdown(f"**{razon or cliente_elegido}**")
                        kc1, kc2, kc3, kc4 = st.columns(4)
                        kc1.metric("Pedidos", n_ped_cli)
                        kc2.metric("Pendiente total", f"{pend_total:,.0f}")
                        kc3.metric("Con sugerencia", int(n_con_cli))
                        kc4.metric("Sin cobertura", int(n_sin_cli))

                        cols_cli = [
                            Columnas.PEDIDO,
                            Columnas.MATERIAL_SOLICITADO,
                            Columnas.DESCRIPCION_SOLICITADA,
                            Columnas.CENTRO_PEDIDO,
                            Columnas.ALMACEN,
                            Columnas.CANTIDAD_PENDIENTE,
                            Columnas.FUENTE,
                            Columnas.CANTIDAD_OFERTAR,
                            Columnas.INV_1030,
                            Columnas.INV_1031,
                            Columnas.INV_1060,
                            Columnas.CONSUMO_DESTINATARIO_12M,
                            Columnas.PROMEDIO_CONSUMO_12M,
                            Columnas.MESES_INVENTARIO,
                            Columnas.BLOQUEADO,
                        ]
                        cols_cli = [c for c in cols_cli if c in df_cli.columns]
                        st.dataframe(
                            df_cli[cols_cli].reset_index(drop=True),
                            use_container_width=True,
                            height=380,
                        )

                        # Resumen de materiales consumidos (del Resumen Sin Sugerencias)
                        if (
                            df_resumen_sin_sugerencias is not None
                            and not df_resumen_sin_sugerencias.empty
                        ):
                            mats_cliente = df_cli[Columnas.MATERIAL_SOLICITADO].unique()
                            df_hist = df_resumen_sin_sugerencias[
                                df_resumen_sin_sugerencias["Material"].isin(
                                    mats_cliente
                                )
                            ]
                            if not df_hist.empty:
                                st.markdown(
                                    "**Histórico de consumo (Resumen Sin Sugerencias)**"
                                )
                                cols_hist = [
                                    c
                                    for c in [
                                        "Centro",
                                        "Almacen",
                                        "Material",
                                        "Descripcion",
                                        "Promedio_Consumo_12M",
                                        "Ultimo_Mes_Consumo",
                                        "Cantidad_Ultimo_Mes",
                                        "Meses_Inventario",
                                    ]
                                    if c in df_hist.columns
                                ]
                                st.dataframe(
                                    df_hist[cols_hist].reset_index(drop=True),
                                    use_container_width=True,
                                    height=250,
                                )

                # ── Descarga ─────────────────────────────────────────────
                st.divider()
                excel_bytes_sugerencias = exportar_reporte_individual(
                    df_todas_sugerencias, "Todas las Sugerencias"
                )
                st.download_button(
                    "📥 Descargar Todas las Sugerencias (Excel)",
                    data=excel_bytes_sugerencias,
                    file_name="Todas_Sugerencias.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_sugerencias",
                )

        # ── Resumen Sin Sugerencias ──────────────────────────────────────
        if (
            generar_resumen_sin_sugerencias_report
            and df_resumen_sin_sugerencias is not None
            and not df_resumen_sin_sugerencias.empty
        ):
            st.session_state.reportes_generados["resumen"] = df_resumen_sin_sugerencias
            with st.expander("✅ Resumen Sin Sugerencias", expanded=True):
                total_pend = df_resumen_sin_sugerencias.get(
                    "Cantidad_Pendiente", pd.Series([0])
                ).sum()
                total_imp = df_resumen_sin_sugerencias.get(
                    "Importe_Pendiente", pd.Series([0])
                ).sum()
                mats = (
                    df_resumen_sin_sugerencias["Material"].nunique()
                    if "Material" in df_resumen_sin_sugerencias.columns
                    else 0
                )
                prom_total = df_resumen_sin_sugerencias.get(
                    "Promedio_Consumo_12M", pd.Series([0])
                ).sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Materiales sin sugerencia", mats)
                c2.metric("Total pendiente", f"{total_pend:,.0f}")
                c3.metric("Total importe", f"${total_imp:,.0f}")
                c4.metric("Consumo promedio 12M", f"{prom_total:,.0f}")

                st.write("**Total pendiente por centro (sin bloqueo):**")
                centros_disp = ["1001", "1003", "1004", "1017", "1018", "1022", "1036"]
                cols_centro = st.columns(min(len(centros_disp), 4))
                for i, centro in enumerate(centros_disp):
                    col_name = f"Pendiente {centro}"
                    if col_name in df_resumen_sin_sugerencias.columns:
                        total = df_resumen_sin_sugerencias[col_name].sum()
                        cols_centro[i % 4].metric(f"Centro {centro}", f"{total:,.0f}")

                st.dataframe(
                    df_resumen_sin_sugerencias.head(10), use_container_width=True
                )
                excel_bytes_resumen = exportar_reporte_individual(
                    df_resumen_sin_sugerencias, "Resumen Sin Sugerencias"
                )
                st.download_button(
                    "📥 Descargar Resumen Sin Sugerencias",
                    data=excel_bytes_resumen,
                    file_name="Resumen_Sin_Sugerencias.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_resumen",
                )

        # ── Sugerencias desde Reporte de Consumo (nuevo reporte) ────────
        if (
            generar_sug_consumo_report
            and df_sug_consumo is not None
            and not df_sug_consumo.empty
        ):
            st.session_state.reportes_generados["sug_consumo"] = df_sug_consumo
            with st.expander(
                "✅ Sugerencias desde Reporte de Consumo", expanded=True
            ):
                # ── KPIs ─────────────────────────────────────────────────
                # Usar nombres reales del nuevo formato (sin Columnas.*)
                _col_fuente_rc = "Fuente"
                _col_mat_rc    = "Material"
                _col_dest_rc   = "Destinatario"

                n_con_rc  = int((df_sug_consumo[_col_fuente_rc] != "").sum())
                n_sin_rc  = int((df_sug_consumo[_col_fuente_rc] == "").sum())
                n_mat_rc  = int(df_sug_consumo[_col_mat_rc].nunique())
                n_dest_rc = int(df_sug_consumo[_col_dest_rc].nunique())

                kc1, kc2, kc3, kc4 = st.columns(4)
                kc1.metric("Materiales únicos",    f"{n_mat_rc:,}")
                kc2.metric("Destinatarios únicos", f"{n_dest_rc:,}")
                kc3.metric("Con sugerencia",        f"{n_con_rc:,}")
                kc4.metric("Sin cobertura",         f"{n_sin_rc:,}")

                # ── Sub-pestañas ──────────────────────────────────────────
                tab_rc_sug, tab_rc_sin = st.tabs(
                    ["📦 Sugerencias activas", "⚠️ Sin cobertura"]
                )

                with tab_rc_sug:
                    st.caption(
                        "Registros del Reporte de Consumo con al menos una fuente disponible"
                    )
                    col_rc1, col_rc2 = st.columns([2, 3])
                    with col_rc1:
                        fuentes_rc = sorted(
                            df_sug_consumo[_col_fuente_rc]
                            .replace("", "Sin sugerencia")
                            .unique()
                        )
                        fuente_rc_sel = st.selectbox(
                            "Fuente",
                            ["Todas"] + list(fuentes_rc),
                            key="rc_fuente",
                        )
                    with col_rc2:
                        busqueda_rc = st.text_input(
                            "Buscar material o descripción", key="rc_buscar"
                        )

                    df_rc_con = df_sug_consumo[
                        df_sug_consumo[_col_fuente_rc] != ""
                    ].copy()
                    if fuente_rc_sel != "Todas":
                        df_rc_con = df_rc_con[
                            df_rc_con[_col_fuente_rc] == fuente_rc_sel
                        ]
                    if busqueda_rc:
                        mask_rc = (
                            df_rc_con["Material"]
                            .astype(str)
                            .str.contains(busqueda_rc, case=False, na=False)
                        ) | (
                            df_rc_con["Texto Material"]
                            .astype(str)
                            .str.contains(busqueda_rc, case=False, na=False)
                        )
                        df_rc_con = df_rc_con[mask_rc]

                    # Columnas de visualización usando los nombres reales del output
                    cols_rc_vista = [
                        "Centro",
                        "Destinatario",
                        "Razón Social",
                        "Material",
                        "Texto Material",
                        "Consumo_promedio_mensual",
                        "Ultima_compra_cliente",
                        "Fuente",
                        "Material sugerido",
                        "Descripción sugerida",
                        "Centro sugerido",
                        "Almacén sugerido",
                        "Disponible",
                        "Lote",
                        "Fecha de Caducidad",
                        "Inv 1030",
                        "Inv 1031",
                        "Inv 1060",
                        "Promedio_Consumo_12M",
                        "Meses_Inventario",
                    ]
                    cols_rc_vista = [c for c in cols_rc_vista if c in df_rc_con.columns]
                    st.dataframe(
                        df_rc_con[cols_rc_vista].reset_index(drop=True),
                        use_container_width=True,
                        height=420,
                    )
                    st.caption(f"{len(df_rc_con):,} líneas mostradas")

                with tab_rc_sin:
                    st.caption(
                        "Registros del Reporte de Consumo sin ninguna fuente disponible"
                    )
                    df_rc_sin = df_sug_consumo[
                        df_sug_consumo[_col_fuente_rc] == ""
                    ].copy()
                    if df_rc_sin.empty:
                        st.success(
                            "¡Todos los registros de consumo tienen al menos una sugerencia!"
                        )
                    else:
                        ks1, ks2 = st.columns(2)
                        ks1.metric(
                            "Materiales sin cobertura",
                            df_rc_sin["Material"].nunique(),
                        )
                        ks2.metric(
                            "Destinatarios sin cobertura",
                            df_rc_sin["Destinatario"].nunique(),
                        )
                        cols_rc_sin = [
                            "Centro",
                            "Destinatario",
                            "Material",
                            "Texto Material",
                            "Consumo_promedio_mensual",
                            "Ultima_compra_cliente",
                            "Inv 1030",
                            "Inv 1031",
                            "Promedio_Consumo_12M",
                        ]
                        cols_rc_sin = [c for c in cols_rc_sin if c in df_rc_sin.columns]
                        st.dataframe(
                            df_rc_sin[cols_rc_sin].reset_index(drop=True),
                            use_container_width=True,
                            height=350,
                        )

                st.divider()
                excel_bytes_sug_consumo = exportar_reporte_individual(
                    df_sug_consumo, "Sug Reporte Consumo"
                )
                st.download_button(
                    "📥 Descargar Sugerencias desde Consumo (Excel)",
                    data=excel_bytes_sug_consumo,
                    file_name="Sugerencias_Reporte_Consumo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_sug_consumo",
                )

        # ── Descarga combinada ───────────────────────────────────────────
        reportes_disp = [
            generar_reporte_consumo_report
            and df_reporte_consumo is not None
            and not df_reporte_consumo.empty,
            generar_todas_sugerencias_report
            and df_todas_sugerencias is not None
            and not df_todas_sugerencias.empty,
            generar_resumen_sin_sugerencias_report
            and df_resumen_sin_sugerencias is not None
            and not df_resumen_sin_sugerencias.empty,
            generar_sug_consumo_report
            and df_sug_consumo is not None
            and not df_sug_consumo.empty,
        ]
        if any(reportes_disp):
            st.divider()
            st.subheader("📦 Descargar Todos los Reportes")
            n_rep = sum(reportes_disp)
            excel_bytes_completo = exportar_a_excel(
                df_todas_sugerencias if generar_todas_sugerencias_report else None,
                (
                    df_resumen_sin_sugerencias
                    if generar_resumen_sin_sugerencias_report
                    else None
                ),
                df_reporte_consumo if generar_reporte_consumo_report else None,
                df_sug_consumo if generar_sug_consumo_report else None,
            )
            label_combo = f"📦 Descargar {n_rep} reporte{'s' if n_rep > 1 else ''} en un solo Excel"
            fname_combo = (
                "Reporte_Completo.xlsx" if n_rep >= 4 else f"Reporte_{n_rep}_hojas.xlsx"
            )
            st.download_button(
                label=label_combo,
                data=excel_bytes_completo,
                file_name=fname_combo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_completo",
            )
        else:
            st.warning("No se generaron datos para exportar")

    except Exception as e:
        st.error(f"Error al procesar los archivos: {str(e)}")
        logger.error(f"Error detallado: {str(e)}", exc_info=True)

else:
    info_text = """
    ### 📌 Instrucciones para cargar los archivos:
    
    1. **Archivo con Seg pedidos** - Contiene la hoja 'Seg pedidos' o 'sheets1' con los pedidos a procesar
    2. **Archivo con Inventario** - Contiene la hoja 'Inventario' o 'sheets1' con los datos de inventario
    3. **Archivo con hojas externas** - Contiene las hojas: Corta caducidad, Lento mov, Cosmopark, Sustituto, PNC, Caduco
    
    ⚠️ **Nota:** Para la hoja de Inventario, se realizará automáticamente el cálculo:
    **"Libre Utilización" - "Entrega a cliente" = Inventario ajustado**
    
    ### 📊 **NOVEDADES en el Reporte "Resumen Sin Sugerencias":**
    1. ✅ **Nueva columna:** "Promedio_Consumo_12M" - Consumo promedio de últimos 12 meses desde datos de facturación
    2. ✅ **Nuevas columnas agregadas:** 
       - "Ultimo_Mes_Consumo" (MM/AAAA)
       - "Penultimo_Mes_Consumo" (MM/AAAA)
       - "Cantidad_Ultimo_Mes"
       - "Cantidad_Penultimo_Mes"
       - "Meses_Inventario" (Inventario total / Consumo promedio)
    3. ✅ **Columnas eliminadas:** "Inv 1001", "Inv 1003", "Inv 1004", "Inv 1017", "Inv 1018", "Inv 1022", "Inv 1036"
    4. ✅ **Nuevas columnas agregadas:** "Pendiente 1001", "Pendiente 1003", etc.
    5. ✅ **Solo incluye** pedidos sin estatus de bloqueo en la columna "Bloqueado"
    """

    if generar_reporte_consumo_report or generar_sug_consumo_report:
        info_text += """
    4. **Archivo de Facturación** - Contiene la hoja 'Facturacion' o 'sheets1' con datos históricos de facturación
       • Columnas requeridas: Solicitante, Razón Social, Destinatario, Fecha, Factura, Doc. Comerc. Ant,
         Material, Texto Material, Cantidad, UM, Importe, Centro, Almacén, Doc. Ventas, Gpo. Vdor., Grp. Cliente
       • **IMPORTANTE:** Ahora también se usa para calcular estadísticas de consumo en "Resumen Sin Sugerencias"
        """

    if generar_sug_consumo_report:
        info_text += """
    ### 🔎 **Nuevo Reporte: "Sugerencias desde Reporte de Consumo"**
    - Aplica la misma lógica de sugerencias que "Todas las Sugerencias", pero sobre los registros
      del **Reporte de Consumo** (histórico de facturación por Destinatario/Material).
    - Requiere tener activo también "Generar Reporte de Consumo".
    - Genera una hoja independiente **"Sug Reporte Consumo"** en el Excel combinado.
    - Mapeo de columnas: Destinatario → Destinatario, Material → Material,
      Centro → Centro, Consumo_promedio_mensual → Pendiente de referencia,
      Cantidad ultima → Cantidad, Precio_unitario_ultima → Precio.
        """

    st.info(info_text)
