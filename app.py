import streamlit as st
import io
import zipfile
from typing import List, Dict, Optional
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

        # Filtrar solo datos válidos
        df_valido = df_facturacion[
            (df_facturacion["Fecha"].notna())
            & (df_facturacion["Cantidad"] > 0)
            & (df_facturacion["Importe"] > 0)
        ].copy()

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

    # Filtrar en una sola operación
    mask_valido = (df_facturacion["Cantidad"] > 0) & (df_facturacion["Importe"] > 0)
    df_facturacion = df_facturacion[mask_valido].copy()

    if df_facturacion.empty:
        progress_bar.empty()
        status_text.empty()
        return pd.DataFrame()

    # Crear columnas auxiliares para cálculos rápidos (vectorizado)
    df_facturacion["AñoMes"] = df_facturacion["Fecha"].dt.to_period("M")
    df_facturacion["PrecioUnitario"] = (
        df_facturacion["Importe"] / df_facturacion["Cantidad"]
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
            cantidad_total_historico=("Cantidad", "sum"),
            fecha_min_historico=("Fecha", "min"),
            fecha_max_historico=("Fecha", "max"),
            meses_con_factura=("AñoMes", "nunique"),
            count_facturas=("Fecha", "count"),
        )
        .reset_index()
    )

    # Calcular precios por grupo (usando todos los datos)
    df_precios_grouped = (
        df_facturacion.groupby(["Solicitante", "Destinatario", "Material"])
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
    for almacen in ["1030", "1031", "1032"]:
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

    if centro_pedido in inventario_centro_almacen:
        almacenes = inventario_centro_almacen[centro_pedido]
        inv_1030 = almacenes.get("1030", 0)
        inv_1031 = almacenes.get("1031", 0)
        inv_1032 = almacenes.get("1032", 0)

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

    if centro_pedido in inventario_centro_almacen:
        almacenes = inventario_centro_almacen[centro_pedido]
        inv_1030 = almacenes.get("1030", 0)
        inv_1031 = almacenes.get("1031", 0)
        inv_1032 = almacenes.get("1032", 0)

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

                                # Formatear fecha si es necesario
                                if pd.notnull(fecha_cad):
                                    try:
                                        fecha_cad = pd.to_datetime(fecha_cad).strftime(
                                            "%d/%m/%Y"
                                        )
                                    except:
                                        fecha_cad = str(fecha_cad)
                                else:
                                    fecha_cad = ""

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

                                # Formatear fecha si es necesario
                                if pd.notnull(fecha_cad):
                                    try:
                                        fecha_cad = pd.to_datetime(fecha_cad).strftime(
                                            "%d/%m/%Y"
                                        )
                                    except:
                                        fecha_cad = str(fecha_cad)
                                else:
                                    fecha_cad = ""

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

                # Usar la nueva función para calcular el disponible según la fuente y lote específico
                disponible_fuente = obtener_disponible_por_fuente(
                    fuente=fuente,
                    material=material_solicitado,
                    centro=centro,
                    almacen=almacen,
                    df_fuente=df_fuente,
                    inventario_df=inventario_df,
                    lote=lote,  # Pasamos el lote específico
                )

                # En la función buscar_sugerencias_exactas (línea ~1580):
                if pd.notnull(fecha_cad):
                    try:
                        # Especificar dayfirst=True para formato dd/mm/aaaa
                        if isinstance(fecha_cad, str):
                            fecha_cad = pd.to_datetime(
                                fecha_cad, dayfirst=True, errors="coerce"
                            )
                        if pd.notnull(fecha_cad):
                            fecha_cad = fecha_cad.strftime("%d/%m/%Y")
                        else:
                            fecha_cad = ""
                    except Exception:
                        fecha_cad = str(fecha_cad)
                else:
                    fecha_cad = ""

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
# Actualizar generar_todas_sugerencias
# =========================
def generar_todas_sugerencias(
    pedidos_df: pd.DataFrame,
    hojas_externas: Dict[str, pd.DataFrame],
    fuentes_activas: List[str],
    inventario_df: pd.DataFrame,
) -> pd.DataFrame:
    """Genera todas las sugerencias para todos los pedidos, incluyendo línea sin sugerencia"""
    todas_sugerencias = []

    # Crear barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_pedidos = len(pedidos_df)

    # Pre-cachear datos de inventario para acceso rápido
    inventario_cache = {}
    if inventario_df is not None and not inventario_df.empty:
        for _, row in inventario_df.iterrows():
            key = f"{row['Centro']}_{row['Material']}_{row['Almacén']}"
            inventario_cache[key] = {
                "libre": float(row.get("Libre Utilización", 0)),
                "transito": float(row.get("Cant. en Tránsito", 0)),
            }

    for i, (_, pedido) in enumerate(pedidos_df.iterrows()):
        # Actualizar barra de progreso
        progress = (i + 1) / total_pedidos
        progress_bar.progress(progress)
        status_text.text(f"Procesando pedido {i+1} de {total_pedidos}")

        # Agregar línea sin sugerencia (fuente vacía)
        linea_sin_sugerencia = crear_linea_sin_sugerencia(pedido, inventario_df)
        todas_sugerencias.append(linea_sin_sugerencia)

        # Buscar sugerencias
        sugerencias_pedido = buscar_sugerencias_exactas(
            pedido, hojas_externas, fuentes_activas, inventario_df
        )
        todas_sugerencias.extend(sugerencias_pedido)

    # Limpiar barra de progreso
    progress_bar.empty()
    status_text.empty()

    # Crear DataFrame con todas las sugerencias
    if todas_sugerencias:
        df_resultado = pd.DataFrame(todas_sugerencias)

        # Ordenar columnas según el orden solicitado
        columnas_orden = [
            Columnas.GRUPO_CLIENTE,
            Columnas.FECHA,
            Columnas.PEDIDO,
            Columnas.GRUPO_VENDEDOR,
            Columnas.SOLICITANTE,
            Columnas.DESTINATARIO,
            Columnas.RAZON_SOCIAL,
            Columnas.CENTRO_PEDIDO,
            Columnas.ALMACEN,
            Columnas.MATERIAL_SOLICITADO,
            Columnas.MATERIAL_BASE,
            Columnas.DESCRIPCION_SOLICITADA,
            Columnas.CANTIDAD_PEDIDO,
            Columnas.CANTIDAD_PENDIENTE,
            Columnas.CANTIDAD_OFERTAR,
            Columnas.PRECIO,
            Columnas.FUENTE,
            Columnas.MATERIAL_SUGERIDO,
            Columnas.DESCRIPCION_SUGERIDA,
            Columnas.CENTRO_SUGERIDO,
            Columnas.ALMACEN_SUGERIDO,
            Columnas.DISPONIBLE,
            Columnas.LOTE,
            Columnas.FECHA_CADUCIDAD,
            Columnas.CENTRO_INV,
            Columnas.INV_1030,
            Columnas.INV_1031,
            Columnas.INV_1032,
            Columnas.CANT_TRANSITO,
            Columnas.CANT_TRANSITO_1030,
            Columnas.CANT_TRANSITO_1031,
            Columnas.CANT_TRANSITO_1032,
            Columnas.DISP_1031_1030,
            Columnas.DISP_1031_1032,
            Columnas.INV_1001,
            Columnas.INV_1003,
            Columnas.INV_1004,
            Columnas.INV_1017,
            Columnas.INV_1018,
            Columnas.INV_1022,
            Columnas.INV_1036,
            Columnas.BLOQUEADO,
        ]

        # Asegurar que todas las columnas existan
        for col in columnas_orden:
            if col not in df_resultado.columns:
                df_resultado[col] = ""

        return df_resultado[columnas_orden]

    return pd.DataFrame()


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

        # Filtrar solo datos válidos
        df_valido = df_facturacion_procesado[
            (df_facturacion_procesado["Fecha"].notna())
            & (df_facturacion_procesado["Cantidad"] > 0)
        ].copy()

        if df_valido.empty:
            return pd.DataFrame()

        # Calcular últimos 12 meses desde la fecha máxima
        fecha_maxima = df_valido["Fecha"].max()
        fecha_inicio_12m = fecha_maxima - pd.DateOffset(months=12)

        # Filtrar últimos 12 meses para el promedio
        df_ultimos_12m = df_valido[df_valido["Fecha"] >= fecha_inicio_12m].copy()

        # Agrupar por Centro, Material, Almacén para estadísticas
        resultados = []

        for (centro, material, almacen), group in df_valido.groupby(
            ["Centro", "Material", "Almacén"]
        ):
            # Obtener los meses únicos ordenados DESCENDENTE por fecha (numérica)
            # Primero creamos un DataFrame con meses únicos ordenados
            meses_df = group[["MesAno_num", "MesAno_str"]].drop_duplicates()
            meses_df = meses_df.sort_values("MesAno_num", ascending=False)

            meses_unicos = meses_df["MesAno_str"].tolist()

            # Inicializar valores
            ultimo_mes = ""
            penultimo_mes = ""
            cantidad_ultimo_mes = 0
            cantidad_penultimo_mes = 0

            # Obtener último y penúltimo mes (ORDENADOS CRONOLÓGICAMENTE)
            if len(meses_unicos) >= 1:
                ultimo_mes = meses_unicos[0]
                # Sumar todas las cantidades del último mes
                ultimo_mes_num = meses_df.iloc[0]["MesAno_num"]
                cantidad_ultimo_mes = group[group["MesAno_num"] == ultimo_mes_num][
                    "Cantidad"
                ].sum()

            if len(meses_unicos) >= 2:
                penultimo_mes = meses_unicos[1]
                # Sumar todas las cantidades del penúltimo mes
                penultimo_mes_num = meses_df.iloc[1]["MesAno_num"]
                cantidad_penultimo_mes = group[
                    group["MesAno_num"] == penultimo_mes_num
                ]["Cantidad"].sum()

            # Calcular promedio de últimos 12 meses
            group_12m = df_ultimos_12m[
                (df_ultimos_12m["Centro"] == centro)
                & (df_ultimos_12m["Material"] == material)
                & (df_ultimos_12m["Almacén"] == almacen)
            ]

            if not group_12m.empty:
                # Calcular meses únicos en los últimos 12 meses
                # Usar MesAno_num para contar meses únicos correctamente
                meses_12m = group_12m["MesAno_num"].nunique()
                if meses_12m > 0:
                    total_cantidad_12m = group_12m["Cantidad"].sum()
                    promedio_consumo_12m = total_cantidad_12m / meses_12m
                else:
                    promedio_consumo_12m = 0
            else:
                promedio_consumo_12m = 0

            resultados.append(
                {
                    "Centro": centro,
                    "Material": material,
                    "Almacen": almacen,
                    "Promedio_Consumo_12M": round(promedio_consumo_12m, 2),
                    "Ultimo_Mes_Consumo": ultimo_mes,
                    "Penultimo_Mes_Consumo": penultimo_mes,
                    "Cantidad_Ultimo_Mes": cantidad_ultimo_mes,
                    "Cantidad_Penultimo_Mes": cantidad_penultimo_mes,
                }
            )

        return pd.DataFrame(resultados)

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

    # 6. PRECOMPUTAR DATOS DE INVENTARIO PARA CÁLCULOS RÁPIDOS
    inventario_cache = {}
    descripcion_cache = {}
    transito_cache = {}

    if inventario_df is not None and not inventario_df.empty:
        for _, row in inventario_df.iterrows():
            centro = str(row.get("Centro", "")).strip()
            material = str(row.get("Material", "")).strip()
            almacen = str(row.get("Almacén", "")).strip()
            libre = float(row.get("Libre Utilización", 0))
            transito = float(row.get("Cant. en Tránsito", 0))
            descripcion = str(row.get("Descripción", "")).strip()

            key_inv = f"{centro}_{material}_{almacen}"
            inventario_cache[key_inv] = libre

            key_desc = f"{centro}_{material}_{almacen}"
            if key_desc not in descripcion_cache and descripcion:
                descripcion_cache[key_desc] = descripcion

            key_trans = f"{centro}_{material}"
            if key_trans not in transito_cache:
                transito_cache[key_trans] = {"1030": 0, "1031": 0, "1032": 0}
            if almacen in ["1030", "1031", "1032"]:
                transito_cache[key_trans][almacen] = (
                    transito_cache[key_trans].get(almacen, 0) + transito
                )

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

    # 9. AGREGAR DATOS DE INVENTARIO ESPECÍFICOS POR ALMACÉN
    def obtener_datos_inventario_resumen(row):
        centro = row["Centro"]
        almacen = row["Almacen"]
        material = row["Material"]

        # Obtener inventario por almacén específico
        inv_1030 = inventario_cache.get(f"{centro}_{material}_1030", 0)
        inv_1031 = inventario_cache.get(f"{centro}_{material}_1031", 0)
        inv_1032 = inventario_cache.get(f"{centro}_{material}_1032", 0)

        # Obtener tránsito para el almacén específico
        transito_key = f"{centro}_{material}"
        transito_almacen = transito_cache.get(transito_key, {}).get(almacen, 0)

        # Disponible en centro 1031 para almacenes 1030 y 1032
        disp_1031_1030 = inventario_cache.get(f"1031_{material}_1030", 0)
        disp_1031_1032 = inventario_cache.get(f"1031_{material}_1032", 0)

        return pd.Series(
            [
                inv_1030,
                inv_1031,
                inv_1032,
                transito_almacen,
                disp_1031_1030,
                disp_1031_1032,
            ]
        )

    columnas_inventario_resumen = [
        "Inv 1030",
        "Inv 1031",
        "Inv 1032",
        "Cant. en Tránsito",
        "Disponible 1031-1030",
        "Disponible 1031-1032",
    ]

    grouped[columnas_inventario_resumen] = grouped.apply(
        obtener_datos_inventario_resumen, axis=1, result_type="expand"
    )

    # 10. CALCULAR MESES DE INVENTARIO
    def calcular_meses_inventario(row):
        # Calcular inventario total en el centro para el material
        inv_total = (
            row.get("Inv 1030", 0) + row.get("Inv 1031", 0) + row.get("Inv 1032", 0)
        )

        # Calcular meses de inventario
        consumo_promedio = row.get("Promedio_Consumo_12M", 0)
        if consumo_promedio > 0:
            return round(inv_total / consumo_promedio, 2)
        else:
            return 0 if inv_total == 0 else 999  # Si hay inventario pero no consumo

    grouped["Meses_Inventario"] = grouped.apply(calcular_meses_inventario, axis=1)

    # 11. CALCULAR PENDIENTE POR CENTRO SIN BLOQUEO - VERSIÓN CORREGIDA
    pendiente_por_centro_dict = None
    if df_todas_sugerencias is not None and not df_todas_sugerencias.empty:
        pendiente_por_centro_dict = calcular_pendiente_por_centro_sin_bloqueo(
            df_todas_sugerencias
        )

    # 12. AGREGAR PENDIENTE POR CENTRO - VERSIÓN CORREGIDA
    centros_interes = ["1001", "1003", "1004", "1017", "1018", "1022", "1036"]

    if pendiente_por_centro_dict:
        for centro in centros_interes:
            if (
                centro in pendiente_por_centro_dict
                and pendiente_por_centro_dict[centro]
            ):
                # Crear diccionario para este centro
                centro_dict = pendiente_por_centro_dict[centro]

                # **CORRECCIÓN:** Solo asignar si el Centro de la fila coincide EXACTAMENTE
                # y si existe en el diccionario
                def asignar_pendiente(row, centro_dict, centro):
                    # Verificar que el centro de la fila sea el mismo que estamos procesando
                    if str(row["Centro"]).strip() != str(centro).strip():
                        return 0

                    # Crear la clave para buscar
                    clave = f"{row['Material']}_{row['Almacen']}"

                    # Buscar en el diccionario
                    if clave in centro_dict:
                        return centro_dict[clave]
                    else:
                        return 0

                grouped[f"Pendiente {centro}"] = grouped.apply(
                    lambda row: asignar_pendiente(row, centro_dict, centro), axis=1
                )
            else:
                grouped[f"Pendiente {centro}"] = 0
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

if generar_reporte_consumo_report:
    archivo_facturacion = st.file_uploader(
        "4. Archivo con pestaña 'Facturacion' o 'sheets1' (Excel)",
        type=["xlsx", "xls"],
        key="facturacion",
    )
else:
    archivo_facturacion = None
    st.info(
        "Para cargar el archivo de facturación, active 'Generar Reporte de Consumo' en la barra lateral"
    )

# Verificar que se hayan subido los 3 archivos
if (
    archivo_principal
    and archivo_inventario
    and archivo_externas
    and (
        not generar_reporte_consumo_report
        or (generar_reporte_consumo_report and archivo_facturacion)
    )
):

    with st.spinner("Procesando archivos..."):
        try:
            # Cargar archivo principal

            xls_principal = pd.ExcelFile(archivo_principal)

            # ------------------------------
            # Detectar hoja principal (Seg pedidos / sheets)
            # ------------------------------
            sheet_map = {s.strip().casefold(): s for s in xls_principal.sheet_names}
            hoja_pedidos = None

            # 1) Por nombre (case-insensitive)
            for candidato in ["seg pedidos", "sheets1"]:
                if candidato in sheet_map:
                    hoja_pedidos = sheet_map[candidato]
                    break

            # 2) Si no coincide por nombre, detectar por columnas mínimas
            if hoja_pedidos is None:
                columnas_minimas = {
                    "Pedido",
                    "Material",
                    "Centro",
                }  # ajusta si tu SAP trae otras fijas
                for sh in xls_principal.sheet_names:
                    try:
                        cols = set(pd.read_excel(xls_principal, sh, nrows=0).columns)
                        if columnas_minimas.issubset(cols):
                            hoja_pedidos = sh
                            break
                    except Exception:
                        pass

            if hoja_pedidos is None:
                st.error(
                    "El archivo principal debe contener la hoja 'Seg pedidos' o 'sheets1' "
                    "o una hoja con columnas mínimas: Pedido, Material, Centro.\n"
                    f"Hojas encontradas: {xls_principal.sheet_names}"
                )
                st.stop()

            pedidos_df = pd.read_excel(xls_principal, hoja_pedidos)

            # ------------------------------
            # Normalizar columnas del archivo principal
            # ------------------------------
            pedidos_df.columns = [
                col.replace("Almacen", "Almacén").replace("Almaçen", "Almacén")
                for col in pedidos_df.columns
            ]

            # ------------------------------
            # Normalizar "Gpo.Vdor."
            # ------------------------------
            col_gpo_vdor = encontrar_columna_por_patron(
                pedidos_df,
                patrones=[
                    "gpo.vdor",
                    "gpo. vdor",
                    "gpo vdor",
                    "grupo vendedor",
                    "gpo vendedor",
                    "vdor",
                ],
            )

            if "Gpo.Vdor." not in pedidos_df.columns:
                if col_gpo_vdor:
                    pedidos_df["Gpo.Vdor."] = pedidos_df[col_gpo_vdor]
                else:
                    pedidos_df["Gpo.Vdor."] = ""

            # Limpieza (por si viene numérico/NaN)
            pedidos_df["Gpo.Vdor."] = (
                pedidos_df["Gpo.Vdor."]
                .astype(str)
                .str.strip()
                .replace({"nan": "", "None": ""})
            )

            # ------------------------------
            # Normalizar IDs
            # ------------------------------
            for col in ["Centro", "Material", "Almacén"]:
                if col in pedidos_df.columns:
                    pedidos_df[col] = normalizar_ids(pedidos_df[col])

            st.success(
                f"✅ Archivo principal procesado: {len(pedidos_df)} pedidos cargados"
            )

            # ------------------------------------------------------------------
            # 2. Procesar archivo de inventario (con cálculo especial)
            # ------------------------------------------------------------------
            st.subheader("📦 Procesando archivo de inventario...")
            xls_inventario = pd.ExcelFile(archivo_inventario)

            # Buscar hoja de inventario
            hoja_inventario = None
            for hoja in xls_inventario.sheet_names:
                if "inventario" in hoja.lower() or "sheets1" in hoja.lower():
                    hoja_inventario = hoja
                    break

            if hoja_inventario is None:
                # Intentar con la primera hoja
                hoja_inventario = xls_inventario.sheet_names[0]
                st.warning(f"Usando hoja '{hoja_inventario}' como inventario")

            df_inventario_raw = pd.read_excel(xls_inventario, hoja_inventario)

            # Aplicar procesamiento especial con cálculo
            inventario_df = procesar_hoja_inventario_ajustada(df_inventario_raw)

            if not inventario_df.empty:
                st.success(f"✅ Inventario procesado: {len(inventario_df)} registros")
                st.sidebar.write(
                    f"**Materiales en inventario:** {inventario_df['Material'].nunique()}"
                )
            else:
                st.warning("El archivo de inventario está vacío o no se pudo procesar")

            # ------------------------------------------------------------------
            # 3. Procesar archivo con hojas externas
            # ------------------------------------------------------------------
            st.subheader("📚 Procesando archivo con hojas externas...")
            xls_externas = pd.ExcelFile(archivo_externas)
            hojas_externas = {}

            # Lista de hojas a procesar (excluyendo posibles hojas de inventario)
            hojas_a_procesar = [
                hoja
                for hoja in xls_externas.sheet_names
                if "inventario" not in hoja.lower()
            ]

            for hoja in hojas_a_procesar:
                if hoja in fuentes_disponibles:
                    df_hoja = pd.read_excel(xls_externas, hoja)

                    if modo_depuracion:
                        st.write(f"**Hoja '{hoja}'**: {len(df_hoja)} filas")
                        st.write(f"Columnas: {df_hoja.columns.tolist()}")

                    # Solo procesar hojas externas si se va a generar el reporte
                    if generar_todas_sugerencias_report:
                        hojas_externas[hoja] = procesar_hoja_externa(df_hoja, hoja)
                        st.write(f"  ✓ {hoja}: {len(hojas_externas[hoja])} registros")

            st.success(
                f"✅ Archivo externo procesado: {len(hojas_externas)} hojas cargadas"
            )

            # ------------------------------------------------------------------
            # INICIALIZAR CACHE PARA DATOS PROCESADOS
            # ------------------------------------------------------------------
            # EN LA SECCIÓN DE INICIALIZACIÓN DEL CACHE (aproximadamente línea 2030-2035):
            if "cache_inicializado" not in st.session_state:
                st.session_state.cache_inicializado = True
                st.session_state.cache_pedidos = None
                st.session_state.cache_inventario = None
                st.session_state.cache_externas = None
                st.session_state.cache_facturacion = None

            # Inicializar la variable fuera del bloque condicional
            df_facturacion_procesado = None  # ← AÑADIR ESTA LÍNEA

            # Opción para usar cache
            usar_cache = st.checkbox(
                "Usar cache de datos procesados (acelera reprocesamiento)", value=True
            )

            # Si el usuario quiere usar cache Y tenemos datos cacheados
            if usar_cache and st.session_state.cache_pedidos is not None:
                # Usar datos cacheados
                pedidos_df = st.session_state.cache_pedidos
                inventario_df = st.session_state.cache_inventario
                hojas_externas = st.session_state.cache_externas

                # Solo usar cache de facturación si existe
                if st.session_state.cache_facturacion is not None:
                    df_facturacion_procesado = (
                        st.session_state.cache_facturacion
                    )  # ← CORREGIDO

                st.success("✓ Usando datos cacheados de ejecución anterior")

                # Mostrar estadísticas de cache
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pedidos cacheados", len(pedidos_df))
                with col2:
                    st.metric("Inventario cacheado", len(inventario_df))
                with col3:
                    st.metric("Hojas externas cacheadas", len(hojas_externas))
            else:
                # Guardar en cache para futuras ejecuciones
                st.session_state.cache_pedidos = pedidos_df
                st.session_state.cache_inventario = inventario_df
                st.session_state.cache_externas = hojas_externas

                # Solo guardar facturación si se procesó y la variable está definida
                if df_facturacion_procesado is not None:
                    st.session_state.cache_facturacion = df_facturacion_procesado

                # Solo guardar facturación si se procesó
                if generar_reporte_consumo_report and archivo_facturacion is not None:
                    try:
                        if (
                            "df_facturacion_procesado" in locals()
                            and df_facturacion_procesado is not None
                        ):
                            st.session_state.cache_facturacion = (
                                df_facturacion_procesado
                            )
                    except:
                        pass

                st.info("✓ Datos guardados en cache para próximas ejecuciones")

            # ------------------------------------------------------------------
            # 4. Procesar archivo de facturación (si está activado)
            # ------------------------------------------------------------------
            df_reporte_consumo = None
            if generar_reporte_consumo_report and archivo_facturacion is not None:
                with st.spinner("Procesando archivo de facturación..."):
                    try:
                        xls_facturacion = pd.ExcelFile(archivo_facturacion)

                        # Buscar hoja de facturación
                        hoja_facturacion = None
                        for hoja in xls_facturacion.sheet_names:
                            if (
                                "facturacion" in hoja.lower()
                                or "sheets1" in hoja.lower()
                            ):
                                hoja_facturacion = hoja
                                break

                        if hoja_facturacion is None:
                            # Intentar con la primera hoja
                            hoja_facturacion = xls_facturacion.sheet_names[0]
                            st.warning(
                                f"Usando hoja '{hoja_facturacion}' como facturación"
                            )

                        df_facturacion_raw = pd.read_excel(
                            xls_facturacion, hoja_facturacion
                        )
                        df_facturacion_procesado = procesar_datos_facturacion(
                            df_facturacion_raw
                        )

                        if not df_facturacion_procesado.empty:
                            st.success(
                                f"✅ Facturación procesada: {len(df_facturacion_procesado)} registros"
                            )

                            # Generar reporte de consumo
                            df_reporte_consumo = generar_reporte_consumo(
                                df_facturacion_procesado
                            )

                            if not df_reporte_consumo.empty:
                                st.success(
                                    f"✅ Reporte de consumo generado: {len(df_reporte_consumo)} materiales"
                                )

                                # Mostrar vista previa del reporte
                                st.subheader("Vista previa del Reporte de Consumo")
                                st.dataframe(df_reporte_consumo.head(), width="stretch")

                                # Estadísticas del reporte
                                st.subheader("Estadísticas del Reporte de Consumo")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Materiales únicos",
                                        df_reporte_consumo["Material"].nunique(),
                                    )
                                with col2:
                                    st.metric(
                                        "Destinatarios",
                                        df_reporte_consumo["Destinatario"].nunique(),
                                    )
                                with col3:
                                    consumo_total = df_reporte_consumo[
                                        "Consumo_promedio_mensual"
                                    ].sum()
                                    st.metric(
                                        "Consumo total mensual", f"{consumo_total:,.0f}"
                                    )
                            else:
                                st.warning("No se pudo generar el reporte de consumo")
                        else:
                            st.warning(
                                "El archivo de facturación está vacío o no se pudo procesar"
                            )

                    except Exception as e:
                        st.error(
                            f"Error al procesar el archivo de facturación: {str(e)}"
                        )
                        logger.error(f"Error en facturación: {str(e)}", exc_info=True)
            elif generar_reporte_consumo_report and archivo_facturacion is None:
                st.warning(
                    "Para generar el reporte de consumo, cargue el archivo de facturación."
                )

                # ------------------------------------------------------------------
            # 5. Generar "Todas las Sugerencias" si está activado
            # ------------------------------------------------------------------
            df_todas_sugerencias = None
            if generar_todas_sugerencias_report:
                with st.spinner("Generando todas las sugerencias..."):
                    try:
                        df_todas_sugerencias = generar_todas_sugerencias(
                            pedidos_df, hojas_externas, fuentes_activas, inventario_df
                        )

                        if (
                            df_todas_sugerencias is not None
                            and not df_todas_sugerencias.empty
                        ):
                            st.success(
                                f"✅ Sugerencias generadas: {len(df_todas_sugerencias)} líneas totales"
                            )

                            # Mostrar estadísticas
                            st.subheader("Estadísticas de Todas las Sugerencias")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Pedidos únicos",
                                    df_todas_sugerencias[Columnas.PEDIDO].nunique(),
                                )
                            with col2:
                                # Contar líneas sin sugerencia
                                sin_sugerencia = df_todas_sugerencias[
                                    df_todas_sugerencias[Columnas.FUENTE] == ""
                                ].shape[0]
                                st.metric("Líneas sin sugerencia", sin_sugerencia)
                            with col3:
                                # Contar líneas con bloqueo
                                con_bloqueo = df_todas_sugerencias[
                                    df_todas_sugerencias[Columnas.BLOQUEADO] != ""
                                ].shape[0]
                                st.metric("Líneas con bloqueo", con_bloqueo)
                        else:
                            st.warning("No se generaron sugerencias")
                    except Exception as e:
                        st.error(f"Error al generar sugerencias: {str(e)}")
                        logger.error(f"Error en sugerencias: {str(e)}", exc_info=True)
            else:
                df_todas_sugerencias = None

            # ------------------------------------------------------------------
            # 6. Generar "Resumen Sin Sugerencias" MODIFICADO con los nuevos requisitos
            # ------------------------------------------------------------------
            df_resumen_sin_sugerencias = None
            if (
                generar_resumen_sin_sugerencias_report
                and df_todas_sugerencias is not None
                and not df_todas_sugerencias.empty
            ):
                with st.spinner("Generando resumen sin sugerencias (MODIFICADO)..."):
                    try:
                        # Usar df_facturacion_procesado si está disponible
                        facturacion_para_resumen = None
                        if (
                            "df_facturacion_procesado" in locals()
                            and df_facturacion_procesado is not None
                        ):
                            facturacion_para_resumen = df_facturacion_procesado

                        # Usar la NUEVA función que incluye los cambios solicitados
                        df_resumen_sin_sugerencias = generar_resumen_sin_sugerencias_optimizado(
                            df_todas_sugerencias,
                            inventario_df,
                            df_todas_sugerencias,  # Pasar también el dataframe completo para calcular pendientes
                            facturacion_para_resumen,
                        )

                        if (
                            df_resumen_sin_sugerencias is not None
                            and not df_resumen_sin_sugerencias.empty
                        ):
                            st.success(
                                f"✅ Resumen MODIFICADO generado: {len(df_resumen_sin_sugerencias)} registros"
                            )

                            # Mostrar las nuevas características
                            st.subheader("Nuevas características del Resumen:")

                            # Verificar columnas agregadas
                            nuevas_columnas_presentes = []
                            if (
                                "Promedio_Consumo_12M"
                                in df_resumen_sin_sugerencias.columns
                            ):
                                nuevas_columnas_presentes.append("Promedio_Consumo_12M")

                            # Verificar columnas de pendiente
                            centros = [
                                "1001",
                                "1003",
                                "1004",
                                "1017",
                                "1018",
                                "1022",
                                "1036",
                            ]
                            for centro in centros:
                                col_name = f"Pendiente {centro}"
                                if col_name in df_resumen_sin_sugerencias.columns:
                                    nuevas_columnas_presentes.append(col_name)

                            # Verificar columnas eliminadas
                            columnas_eliminadas = []
                            for col in [
                                "Inv 1001",
                                "Inv 1003",
                                "Inv 1004",
                                "Inv 1017",
                                "Inv 1018",
                                "Inv 1022",
                                "Inv 1036",
                            ]:
                                if col not in df_resumen_sin_sugerencias.columns:
                                    columnas_eliminadas.append(col)

                            st.write(
                                f"**Columnas agregadas:** {len(nuevas_columnas_presentes)}"
                            )
                            st.write(
                                f"**Columnas eliminadas:** {len(columnas_eliminadas)}"
                            )

                            # Mostrar estadísticas de las nuevas columnas
                            if (
                                "Promedio_Consumo_12M"
                                in df_resumen_sin_sugerencias.columns
                            ):
                                promedio_total = df_resumen_sin_sugerencias[
                                    "Promedio_Consumo_12M"
                                ].sum()
                                st.metric(
                                    "Consumo promedio total (12M)",
                                    f"{promedio_total:,.0f}",
                                )

                            # Mostrar total de pendiente por centro
                            st.write("**Total pendiente por centro (sin bloqueo):**")
                            cols = st.columns(3)
                            for i, centro in enumerate(centros[:3]):
                                col_name = f"Pendiente {centro}"
                                if col_name in df_resumen_sin_sugerencias.columns:
                                    total = df_resumen_sin_sugerencias[col_name].sum()
                                    cols[i].metric(f"Centro {centro}", f"{total:,.0f}")

                            if len(centros) > 3:
                                cols = st.columns(3)
                                for i, centro in enumerate(centros[3:6]):
                                    col_name = f"Pendiente {centro}"
                                    if col_name in df_resumen_sin_sugerencias.columns:
                                        total = df_resumen_sin_sugerencias[
                                            col_name
                                        ].sum()
                                        cols[i].metric(
                                            f"Centro {centro}", f"{total:,.0f}"
                                        )

                            if len(centros) > 6:
                                col_name = f"Pendiente {centros[6]}"
                                if col_name in df_resumen_sin_sugerencias.columns:
                                    total = df_resumen_sin_sugerencias[col_name].sum()
                                    st.metric(f"Centro {centros[6]}", f"{total:,.0f}")
                        else:
                            st.warning("No se pudo generar el resumen modificado")
                    except Exception as e:
                        st.error(f"Error al generar resumen modificado: {str(e)}")
                        logger.error(
                            f"Error en resumen modificado: {str(e)}", exc_info=True
                        )
            else:
                df_resumen_sin_sugerencias = None

            # ------------------------------------------------------------------
            # Generar y mostrar reportes con descargas individuales
            # ------------------------------------------------------------------
            st.header("📊 Reportes Generados")

            # Contenedor para almacenar los reportes generados
            if "reportes_generados" not in st.session_state:
                st.session_state.reportes_generados = {}

            # Generar reporte de consumo si está activado
            if (
                generar_reporte_consumo_report
                and df_reporte_consumo is not None
                and not df_reporte_consumo.empty
            ):
                st.session_state.reportes_generados["consumo"] = df_reporte_consumo

                st.subheader("✅ Reporte de Consumo Listo")
                st.dataframe(df_reporte_consumo.head(), width="stretch")

                # Estadísticas del reporte
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Materiales únicos", df_reporte_consumo["Material"].nunique()
                    )
                with col2:
                    st.metric(
                        "Destinatarios", df_reporte_consumo["Destinatario"].nunique()
                    )
                with col3:
                    consumo_total = df_reporte_consumo["Consumo_promedio_mensual"].sum()
                    st.metric("Consumo total mensual", f"{consumo_total:,.0f}")

                # Botón de descarga individual
                with st.spinner("Preparando descarga del Reporte de Consumo..."):
                    excel_bytes_consumo = exportar_reporte_individual(
                        df_reporte_consumo, "Reporte de Consumo"
                    )

                st.download_button(
                    label="📥 Descargar Reporte de Consumo",
                    data=excel_bytes_consumo,
                    file_name="Reporte_Consumo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_consumo",
                )

            # Generar reporte "Todas las Sugerencias" si está activado
            if (
                generar_todas_sugerencias_report
                and df_todas_sugerencias is not None
                and not df_todas_sugerencias.empty
            ):
                st.session_state.reportes_generados["sugerencias"] = (
                    df_todas_sugerencias
                )

                st.subheader("✅ Todas las Sugerencias Listas")
                st.dataframe(df_todas_sugerencias.head(), width="stretch")

                # Resumen por fuente
                resumen_fuentes = (
                    df_todas_sugerencias.groupby(Columnas.FUENTE)
                    .agg(
                        {
                            Columnas.MATERIAL_SOLICITADO: "nunique",
                            Columnas.CANTIDAD_PENDIENTE: "sum",
                            Columnas.CANTIDAD_OFERTAR: "sum",
                        }
                    )
                    .reset_index()
                )
                resumen_fuentes.columns = [
                    "Fuente",
                    "Materiales Únicos",
                    "Cantidad Pendiente Total",
                    "Cantidad Ofertada Total",
                ]
                st.dataframe(resumen_fuentes, width="stretch")

                # Botón de descarga individual
                with st.spinner("Preparando descarga de Todas las Sugerencias..."):
                    excel_bytes_sugerencias = exportar_reporte_individual(
                        df_todas_sugerencias, "Todas las Sugerencias"
                    )

                st.download_button(
                    label="📥 Descargar Todas las Sugerencias",
                    data=excel_bytes_sugerencias,
                    file_name="Todas_Sugerencias.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_sugerencias",
                )

            # Generar reporte "Resumen Sin Sugerencias" MODIFICADO si está activado
            if (
                generar_resumen_sin_sugerencias_report
                and df_resumen_sin_sugerencias is not None
                and not df_resumen_sin_sugerencias.empty
            ):
                st.session_state.reportes_generados["resumen"] = (
                    df_resumen_sin_sugerencias
                )

                st.subheader("✅ Resumen Sin Sugerencias MODIFICADO Listo")
                st.info(
                    "**Nota:** Este reporte incluye las modificaciones solicitadas:"
                )
                st.write("1. ✅ Columna 'Promedio_Consumo_12M' agregada")
                st.write("2. ✅ Columnas 'Inv 1001, 1003, etc.' eliminadas")
                st.write("3. ✅ Columnas 'Pendiente 1001, 1003, etc.' agregadas")

                st.dataframe(df_resumen_sin_sugerencias.head(), width="stretch")

                # Calcular estadísticas del resumen MODIFICADO
                if "Cantidad" in df_resumen_sin_sugerencias.columns:
                    total_pendiente = df_resumen_sin_sugerencias["Cantidad"].sum()
                else:
                    total_pendiente = 0

                if "Importe" in df_resumen_sin_sugerencias.columns:
                    total_importe = df_resumen_sin_sugerencias["Importe"].sum()
                else:
                    total_importe = 0

                if "Material" in df_resumen_sin_sugerencias.columns:
                    materiales_unicos = df_resumen_sin_sugerencias["Material"].nunique()
                else:
                    materiales_unicos = 0

                if "Promedio_Consumo_12M" in df_resumen_sin_sugerencias.columns:
                    promedio_total = df_resumen_sin_sugerencias[
                        "Promedio_Consumo_12M"
                    ].sum()
                else:
                    promedio_total = 0

                st.subheader(f"Resumen General MODIFICADO:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Materiales sin sugerencia", materiales_unicos)
                with col2:
                    st.metric("Total pendiente", f"{total_pendiente:,.0f}")
                with col3:
                    st.metric("Total importe", f"${total_importe:,.0f}")
                with col4:
                    st.metric("Consumo promedio 12M", f"{promedio_total:,.0f}")

                # Botón de descarga individual
                with st.spinner(
                    "Preparando descarga del Resumen Sin Sugerencias MODIFICADO..."
                ):
                    excel_bytes_resumen = exportar_reporte_individual(
                        df_resumen_sin_sugerencias, "Resumen Sin Sugerencias"
                    )

                st.download_button(
                    label="📥 Descargar Resumen Sin Sugerencias (MODIFICADO)",
                    data=excel_bytes_resumen,
                    file_name="Resumen_Sin_Sugerencias_MODIFICADO.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_resumen",
                )

            # Botón para exportar todos los reportes juntos (si hay al menos uno)
            reportes_disponibles = [
                (
                    generar_reporte_consumo_report
                    and df_reporte_consumo is not None
                    and not df_reporte_consumo.empty
                ),
                (
                    generar_todas_sugerencias_report
                    and df_todas_sugerencias is not None
                    and not df_todas_sugerencias.empty
                ),
                (
                    generar_resumen_sin_sugerencias_report
                    and df_resumen_sin_sugerencias is not None
                    and not df_resumen_sin_sugerencias.empty
                ),
            ]

            if any(reportes_disponibles):
                st.divider()
                st.subheader("📦 Descargar Todos los Reportes")

                with st.spinner("Preparando archivo combinado..."):
                    excel_bytes_completo = exportar_a_excel(
                        (
                            df_todas_sugerencias
                            if generar_todas_sugerencias_report
                            else None
                        ),
                        (
                            df_resumen_sin_sugerencias
                            if generar_resumen_sin_sugerencias_report
                            else None
                        ),
                        df_reporte_consumo if generar_reporte_consumo_report else None,
                    )

                # Determinar nombre del archivo basado en los reportes incluidos
                if sum(reportes_disponibles) == 3:
                    file_name = "Reporte_Completo_MODIFICADO.xlsx"
                    label = "📦 Descargar Excel con todos los reportes (MODIFICADO)"
                elif sum(reportes_disponibles) == 2:
                    file_name = "Reporte_Parcial_MODIFICADO.xlsx"
                    label = "📦 Descargar Excel con reportes disponibles (MODIFICADO)"
                else:
                    file_name = "Reporte_Individual_MODIFICADO.xlsx"
                    label = "📦 Descargar Excel con reporte disponible (MODIFICADO)"

                st.download_button(
                    label=label,
                    data=excel_bytes_completo,
                    file_name=file_name,
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

    if generar_reporte_consumo_report:
        info_text += """
    4. **Archivo de Facturación** - Contiene la hoja 'Facturacion' o 'sheets1' con datos históricos de facturación
       • Columnas requeridas: Solicitante, Razón Social, Destinatario, Fecha, Factura, Doc. Comerc. Ant,
         Material, Texto Material, Cantidad, UM, Importe, Centro, Almacén, Doc. Ventas, Gpo. Vdor., Grp. Cliente
       • **IMPORTANTE:** Ahora también se usa para calcular estadísticas de consumo en "Resumen Sin Sugerencias"
        """

    st.info(info_text)
