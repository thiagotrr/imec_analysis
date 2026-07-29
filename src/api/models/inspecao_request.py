"""Contratos Pydantic de entrada dos endpoints de inspeção de medidor (Task 006, §2).

Contratos **FIXOS** (não gerados em runtime):

- ``LaudoCompletoRequest``: todas as colunas do laudo de aferição presentes hoje
  no dataset de origem (`resultado_laudo_afericao.xlsx`), exceto o target
  ``CODRSTAFER``.
- ``LaudoSinteticoRequest``: apenas as ``retained_feature_columns`` do último
  treino definitivo (`model/preparation_metadata.json`) — o conjunto mínimo
  que o pipeline de pré-processamento/`preprocessing_pipeline.pkl` precisa
  para executar a inferência. Tendência: imutável após o modelo campeão.

``extra="forbid"`` em ambos: campos desconhecidos são rejeitados na validação.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InspecaoMedidorBase(BaseModel):
    """Base comum dos contratos de laudo (§2 do plano)."""

    model_config = ConfigDict(extra="forbid")


# Colunas retidas pelo último treino definitivo (preparation_metadata.json →
# retained_feature_columns). Fonte de verdade do contrato sintético.
RETAINED_FEATURE_COLUMNS: tuple[str, ...] = (
    "VLRENS_CGA_IDU",
    "VLRENS_CGA_PQN",
    "NUMLAUDO",
    "DSCMOD",
    "CODFORN",
    "NUMELM",
    "TENSAONOM",
    "TIPOMDR",
    "TIPO",
    "CODTIP_MED",
    "CODLCD_ITC",
    "VLRLTR",
    "FTRCRC_CGA_NMN_SER",
    "USUARIO_AFER",
    "INDRST_ENS_ISP",
    "INDRST_ENS_EXD",
    "INDRST_ENS_MRC",
    "INDRST_ENS_EXM",
    "INDRST_GER",
    "SIT_LACRE",
    "INDRST_ENS_CRP",
    "CODFTR_CRC",
    "NUM_INVOLUCRO_EQPTO_MDCAO",
    "LEITURA_KWH",
)


class LaudoCompletoRequest(InspecaoMedidorBase):
    """Contrato forte: todas as colunas do laudo (exceto ``CODRSTAFER``)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "NUMOSM": 2501000232,
                "DATAINIAFER": "2025-10-27 13:33:01",
                "DATATERAFER": "2025-10-27 14:14:37",
                "VLRENS_CGA_NMN": -0.02,
                "VLRENS_CGA_CPC": 0.0,
                "VLRENS_CGA_IDU": 0.06,
                "VLRENS_CGA_PQN": 0.0,
                "RESPAFER": 132,
                "OBSAFER": None,
                "CODPRSERV": 301010201,
                "DATALAUDO": "2025-10-28 09:55:00",
                "NUMLAUDO": 2025006988,
                "DSCMOD": "VECTOR 3 PAR TRI BID",
                "CODFORN": 14,
                "NUMELM": 3,
                "NUMFASE": 3,
                "NUMFIO": 4,
                "TENSAONOM": 120,
                "TIPOMDR": "E",
                "TIPO": "T",
                "CODTIP_MED": "D",
                "CODLCD_ITC": 10,
                "VLRDVI_ELM_A": None,
                "VLRDVI_ELM_B": None,
                "VLRDVI_ELM_C": None,
                "DSCRSTAFER": (
                    "O medidor não está de acordo com o Regulamento Técnico Metrológico "
                    "acima referenciado, o TLI (Terminal de Leitura Individual) não foi "
                    "apresentado junto ao medidor, porém não influencia no registro do "
                    "consumo, com defeito em relação as características de fabricação."
                ),
                "VLRLTR": 6116,
                "FTRCRC_CGA_NMN_SER": 1.3,
                "FTRCRC_CGA_NMN_ELM": None,
                "FTRCRC_CGA_IDU": 1.3,
                "FTRCRC_CGA_PQN": 1.3,
                "TPRIFR": 20,
                "TPRSUP": 30,
                "FTRCRC_CGA_NMN_SER_1": 2.6,
                "FTRCRC_CGA_NMN_ELM_1": None,
                "FTRCRC_CGA_IDU_1": 2.6,
                "FTRCRC_CGA_PQN_1": 2.6,
                "TPRIFR_1": 20.0,
                "TPRSUP_1": 30.0,
                "USUARIO_AFER": "RS54",
                "CODPRJ": None,
                "ENSAIO_LAUDO_CORR": None,
                "INDRST_ENS_ISP": "R",
                "INDRST_ENS_EXD": "A",
                "INDRST_ENS_COR": None,
                "INDRST_ENS_MRC": "A",
                "INDRST_ENS_TNS": None,
                "INDRST_ENS_EXM": "A",
                "INDRST_GER": "R",
                "ENSAIO_LAUDO_MESA": 4.0,
                "ENSAIO_LAUDO_TEMPERATURA": 23,
                "ENSAIO_LAUDO_TEMPERATURA2": 26,
                "SIT_LACRE": "APROVADO",
                "IND_LAUDO_EXTERNO": 1.0,
                "INDRST_ENS_CRP": "A",
                "VLRENS_CGA_NMN_1": -0.02,
                "VLRENS_CGA_CPC_1": 0.0,
                "VLRENS_CGA_IDU_1": 0.06,
                "VLRENS_CGA_PQN_1": 0.0,
                "VLRDVI_ELM_A_1": None,
                "VLRDVI_ELM_B_1": None,
                "VLRDVI_ELM_C_1": None,
                "VLRLTR_1": "4859",
                "INDRST_ENS_ISP_1": "R",
                "INDRST_ENS_EXD_1": "A",
                "INDRST_ENS_COR_1": None,
                "INDRST_ENS_MRC_1": "A",
                "INDRST_ENS_TNS_1": None,
                "INDRST_ENS_EXM_1": "A",
                "INDRST_GER_1": "R",
                "INDRST_ENS_CRP_1": "A",
                "VLRLTR_MAN_KWH": None,
                "VLRLTR_MAN_KVARH": None,
                "CODFTR_CRC": 3,
                "CODFTR_CRC_1": 4.0,
                "DAT_ENVIO_AFERICAO": "2025-10-24 09:27:46",
                "DAT_RETORNO_AFERICAO": "2025-10-28 10:35:02",
                "NUM_INVOLUCRO_EQPTO_MDCAO": 164359,
                "NUM_INVOLUCRO": "164359",
                "LEITURA_KWH": "6116",
                "COD_VERIFICADOR_LAUDO": "348414b28",
                "LEITURA_KVARH": "4859",
                "SEQ_NUMLAUDO": 6988,
            }
        },
    )

    NUMOSM: int = Field(..., description="Coluna 'NUMOSM' do laudo de aferição.")
    DATAINIAFER: str = Field(..., description="Coluna 'DATAINIAFER' do laudo de aferição.")
    DATATERAFER: str = Field(..., description="Coluna 'DATATERAFER' do laudo de aferição.")
    VLRENS_CGA_NMN: float | None = Field(default=None, description="Coluna 'VLRENS_CGA_NMN' do laudo de aferição.")
    VLRENS_CGA_CPC: float | None = Field(default=None, description="Coluna 'VLRENS_CGA_CPC' do laudo de aferição.")
    VLRENS_CGA_IDU: float | None = Field(default=None, description="Coluna 'VLRENS_CGA_IDU' do laudo de aferição.")
    VLRENS_CGA_PQN: float | None = Field(default=None, description="Coluna 'VLRENS_CGA_PQN' do laudo de aferição.")
    RESPAFER: int = Field(..., description="Coluna 'RESPAFER' do laudo de aferição.")
    OBSAFER: str | None = Field(default=None, description="Coluna 'OBSAFER' do laudo de aferição.")
    CODPRSERV: int = Field(..., description="Coluna 'CODPRSERV' do laudo de aferição.")
    DATALAUDO: str = Field(..., description="Coluna 'DATALAUDO' do laudo de aferição.")
    NUMLAUDO: int = Field(..., description="Coluna 'NUMLAUDO' do laudo de aferição.")
    DSCMOD: str = Field(..., description="Coluna 'DSCMOD' do laudo de aferição.")
    CODFORN: int = Field(..., description="Coluna 'CODFORN' do laudo de aferição.")
    NUMELM: int = Field(..., description="Coluna 'NUMELM' do laudo de aferição.")
    NUMFASE: int = Field(..., description="Coluna 'NUMFASE' do laudo de aferição.")
    NUMFIO: int = Field(..., description="Coluna 'NUMFIO' do laudo de aferição.")
    TENSAONOM: int = Field(..., description="Coluna 'TENSAONOM' do laudo de aferição.")
    TIPOMDR: str = Field(..., description="Coluna 'TIPOMDR' do laudo de aferição.")
    TIPO: str = Field(..., description="Coluna 'TIPO' do laudo de aferição.")
    CODTIP_MED: str = Field(..., description="Coluna 'CODTIP_MED' do laudo de aferição.")
    CODLCD_ITC: int = Field(..., description="Coluna 'CODLCD_ITC' do laudo de aferição.")
    VLRDVI_ELM_A: float | None = Field(default=None, description="Coluna 'VLRDVI_ELM_A' do laudo de aferição.")
    VLRDVI_ELM_B: float | None = Field(default=None, description="Coluna 'VLRDVI_ELM_B' do laudo de aferição.")
    VLRDVI_ELM_C: float | None = Field(default=None, description="Coluna 'VLRDVI_ELM_C' do laudo de aferição.")
    DSCRSTAFER: str = Field(..., description="Coluna 'DSCRSTAFER' do laudo de aferição.")
    VLRLTR: int = Field(..., description="Coluna 'VLRLTR' do laudo de aferição.")
    FTRCRC_CGA_NMN_SER: float = Field(..., description="Coluna 'FTRCRC_CGA_NMN_SER' do laudo de aferição.")
    FTRCRC_CGA_NMN_ELM: float | None = Field(default=None, description="Coluna 'FTRCRC_CGA_NMN_ELM' do laudo de aferição.")
    FTRCRC_CGA_IDU: float = Field(..., description="Coluna 'FTRCRC_CGA_IDU' do laudo de aferição.")
    FTRCRC_CGA_PQN: float = Field(..., description="Coluna 'FTRCRC_CGA_PQN' do laudo de aferição.")
    TPRIFR: int = Field(..., description="Coluna 'TPRIFR' do laudo de aferição.")
    TPRSUP: int = Field(..., description="Coluna 'TPRSUP' do laudo de aferição.")
    FTRCRC_CGA_NMN_SER_1: float | None = Field(default=None, description="Coluna 'FTRCRC_CGA_NMN_SER_1' do laudo de aferição.")
    FTRCRC_CGA_NMN_ELM_1: float | None = Field(default=None, description="Coluna 'FTRCRC_CGA_NMN_ELM_1' do laudo de aferição.")
    FTRCRC_CGA_IDU_1: float | None = Field(default=None, description="Coluna 'FTRCRC_CGA_IDU_1' do laudo de aferição.")
    FTRCRC_CGA_PQN_1: float | None = Field(default=None, description="Coluna 'FTRCRC_CGA_PQN_1' do laudo de aferição.")
    TPRIFR_1: float | None = Field(default=None, description="Coluna 'TPRIFR_1' do laudo de aferição.")
    TPRSUP_1: float | None = Field(default=None, description="Coluna 'TPRSUP_1' do laudo de aferição.")
    USUARIO_AFER: str = Field(..., description="Coluna 'USUARIO_AFER' do laudo de aferição.")
    CODPRJ: float | None = Field(default=None, description="Coluna 'CODPRJ' do laudo de aferição.")
    ENSAIO_LAUDO_CORR: float | None = Field(default=None, description="Coluna 'ENSAIO_LAUDO_CORR' do laudo de aferição.")
    INDRST_ENS_ISP: str = Field(..., description="Coluna 'INDRST_ENS_ISP' do laudo de aferição.")
    INDRST_ENS_EXD: str | None = Field(default=None, description="Coluna 'INDRST_ENS_EXD' do laudo de aferição.")
    INDRST_ENS_COR: float | None = Field(default=None, description="Coluna 'INDRST_ENS_COR' do laudo de aferição.")
    INDRST_ENS_MRC: str = Field(..., description="Coluna 'INDRST_ENS_MRC' do laudo de aferição.")
    INDRST_ENS_TNS: float | None = Field(default=None, description="Coluna 'INDRST_ENS_TNS' do laudo de aferição.")
    INDRST_ENS_EXM: str = Field(..., description="Coluna 'INDRST_ENS_EXM' do laudo de aferição.")
    INDRST_GER: str = Field(..., description="Coluna 'INDRST_GER' do laudo de aferição.")
    ENSAIO_LAUDO_MESA: float | None = Field(default=None, description="Coluna 'ENSAIO_LAUDO_MESA' do laudo de aferição.")
    ENSAIO_LAUDO_TEMPERATURA: int = Field(..., description="Coluna 'ENSAIO_LAUDO_TEMPERATURA' do laudo de aferição.")
    ENSAIO_LAUDO_TEMPERATURA2: int = Field(..., description="Coluna 'ENSAIO_LAUDO_TEMPERATURA2' do laudo de aferição.")
    SIT_LACRE: str = Field(..., description="Coluna 'SIT_LACRE' do laudo de aferição.")
    IND_LAUDO_EXTERNO: float | None = Field(default=None, description="Coluna 'IND_LAUDO_EXTERNO' do laudo de aferição.")
    INDRST_ENS_CRP: str = Field(..., description="Coluna 'INDRST_ENS_CRP' do laudo de aferição.")
    VLRENS_CGA_NMN_1: float | None = Field(default=None, description="Coluna 'VLRENS_CGA_NMN_1' do laudo de aferição.")
    VLRENS_CGA_CPC_1: float | None = Field(default=None, description="Coluna 'VLRENS_CGA_CPC_1' do laudo de aferição.")
    VLRENS_CGA_IDU_1: float | None = Field(default=None, description="Coluna 'VLRENS_CGA_IDU_1' do laudo de aferição.")
    VLRENS_CGA_PQN_1: float | None = Field(default=None, description="Coluna 'VLRENS_CGA_PQN_1' do laudo de aferição.")
    VLRDVI_ELM_A_1: float | None = Field(default=None, description="Coluna 'VLRDVI_ELM_A_1' do laudo de aferição.")
    VLRDVI_ELM_B_1: float | None = Field(default=None, description="Coluna 'VLRDVI_ELM_B_1' do laudo de aferição.")
    VLRDVI_ELM_C_1: float | None = Field(default=None, description="Coluna 'VLRDVI_ELM_C_1' do laudo de aferição.")
    VLRLTR_1: str | None = Field(default=None, description="Coluna 'VLRLTR_1' do laudo de aferição.")
    INDRST_ENS_ISP_1: str | None = Field(default=None, description="Coluna 'INDRST_ENS_ISP_1' do laudo de aferição.")
    INDRST_ENS_EXD_1: str | None = Field(default=None, description="Coluna 'INDRST_ENS_EXD_1' do laudo de aferição.")
    INDRST_ENS_COR_1: float | None = Field(default=None, description="Coluna 'INDRST_ENS_COR_1' do laudo de aferição.")
    INDRST_ENS_MRC_1: str | None = Field(default=None, description="Coluna 'INDRST_ENS_MRC_1' do laudo de aferição.")
    INDRST_ENS_TNS_1: float | None = Field(default=None, description="Coluna 'INDRST_ENS_TNS_1' do laudo de aferição.")
    INDRST_ENS_EXM_1: str | None = Field(default=None, description="Coluna 'INDRST_ENS_EXM_1' do laudo de aferição.")
    INDRST_GER_1: str | None = Field(default=None, description="Coluna 'INDRST_GER_1' do laudo de aferição.")
    INDRST_ENS_CRP_1: str | None = Field(default=None, description="Coluna 'INDRST_ENS_CRP_1' do laudo de aferição.")
    VLRLTR_MAN_KWH: float | None = Field(default=None, description="Coluna 'VLRLTR_MAN_KWH' do laudo de aferição.")
    VLRLTR_MAN_KVARH: float | None = Field(default=None, description="Coluna 'VLRLTR_MAN_KVARH' do laudo de aferição.")
    CODFTR_CRC: int = Field(..., description="Coluna 'CODFTR_CRC' do laudo de aferição.")
    CODFTR_CRC_1: float | None = Field(default=None, description="Coluna 'CODFTR_CRC_1' do laudo de aferição.")
    DAT_ENVIO_AFERICAO: str = Field(..., description="Coluna 'DAT_ENVIO_AFERICAO' do laudo de aferição.")
    DAT_RETORNO_AFERICAO: str = Field(..., description="Coluna 'DAT_RETORNO_AFERICAO' do laudo de aferição.")
    NUM_INVOLUCRO_EQPTO_MDCAO: int = Field(..., description="Coluna 'NUM_INVOLUCRO_EQPTO_MDCAO' do laudo de aferição.")
    NUM_INVOLUCRO: str = Field(..., description="Coluna 'NUM_INVOLUCRO' do laudo de aferição.")
    LEITURA_KWH: str = Field(..., description="Coluna 'LEITURA_KWH' do laudo de aferição.")
    COD_VERIFICADOR_LAUDO: str | None = Field(default=None, description="Coluna 'COD_VERIFICADOR_LAUDO' do laudo de aferição.")
    LEITURA_KVARH: str | None = Field(default=None, description="Coluna 'LEITURA_KVARH' do laudo de aferição.")
    SEQ_NUMLAUDO: int = Field(..., description="Coluna 'SEQ_NUMLAUDO' do laudo de aferição.")


class LaudoSinteticoRequest(InspecaoMedidorBase):
    """Contrato fixo: apenas as features retidas pelo último treino definitivo.

    Espelha ``RETAINED_FEATURE_COLUMNS`` / ``preparation_metadata.json`` —
    o mínimo necessário para o ``preprocessing_pipeline.pkl`` ser executável.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "VLRENS_CGA_IDU": 0.06,
                "VLRENS_CGA_PQN": 0.0,
                "NUMLAUDO": 2025006988,
                "DSCMOD": "VECTOR 3 PAR TRI BID",
                "CODFORN": 14,
                "NUMELM": 3,
                "TENSAONOM": 120,
                "TIPOMDR": "E",
                "TIPO": "T",
                "CODTIP_MED": "D",
                "CODLCD_ITC": 10,
                "VLRLTR": 6116,
                "FTRCRC_CGA_NMN_SER": 1.3,
                "USUARIO_AFER": "RS54",
                "INDRST_ENS_ISP": "R",
                "INDRST_ENS_EXD": "A",
                "INDRST_ENS_MRC": "A",
                "INDRST_ENS_EXM": "A",
                "INDRST_GER": "R",
                "SIT_LACRE": "APROVADO",
                "INDRST_ENS_CRP": "A",
                "CODFTR_CRC": 3,
                "NUM_INVOLUCRO_EQPTO_MDCAO": 164359,
                "LEITURA_KWH": "6116",
            }
        },
    )

    VLRENS_CGA_IDU: float | None = Field(default=None, description="Feature retida do laudo ('VLRENS_CGA_IDU').")
    VLRENS_CGA_PQN: float | None = Field(default=None, description="Feature retida do laudo ('VLRENS_CGA_PQN').")
    NUMLAUDO: int = Field(..., description="Feature retida do laudo ('NUMLAUDO').")
    DSCMOD: str = Field(..., description="Feature retida do laudo ('DSCMOD').")
    CODFORN: int = Field(..., description="Feature retida do laudo ('CODFORN').")
    NUMELM: int = Field(..., description="Feature retida do laudo ('NUMELM').")
    TENSAONOM: int = Field(..., description="Feature retida do laudo ('TENSAONOM').")
    TIPOMDR: str = Field(..., description="Feature retida do laudo ('TIPOMDR').")
    TIPO: str = Field(..., description="Feature retida do laudo ('TIPO').")
    CODTIP_MED: str = Field(..., description="Feature retida do laudo ('CODTIP_MED').")
    CODLCD_ITC: int = Field(..., description="Feature retida do laudo ('CODLCD_ITC').")
    VLRLTR: int = Field(..., description="Feature retida do laudo ('VLRLTR').")
    FTRCRC_CGA_NMN_SER: float = Field(..., description="Feature retida do laudo ('FTRCRC_CGA_NMN_SER').")
    USUARIO_AFER: str = Field(..., description="Feature retida do laudo ('USUARIO_AFER').")
    INDRST_ENS_ISP: str = Field(..., description="Feature retida do laudo ('INDRST_ENS_ISP').")
    INDRST_ENS_EXD: str | None = Field(default=None, description="Feature retida do laudo ('INDRST_ENS_EXD').")
    INDRST_ENS_MRC: str = Field(..., description="Feature retida do laudo ('INDRST_ENS_MRC').")
    INDRST_ENS_EXM: str = Field(..., description="Feature retida do laudo ('INDRST_ENS_EXM').")
    INDRST_GER: str = Field(..., description="Feature retida do laudo ('INDRST_GER').")
    SIT_LACRE: str = Field(..., description="Feature retida do laudo ('SIT_LACRE').")
    INDRST_ENS_CRP: str = Field(..., description="Feature retida do laudo ('INDRST_ENS_CRP').")
    CODFTR_CRC: int = Field(..., description="Feature retida do laudo ('CODFTR_CRC').")
    NUM_INVOLUCRO_EQPTO_MDCAO: int = Field(..., description="Feature retida do laudo ('NUM_INVOLUCRO_EQPTO_MDCAO').")
    LEITURA_KWH: str = Field(..., description="Feature retida do laudo ('LEITURA_KWH').")


def validate_laudo_completo_row(row: dict[str, object]) -> LaudoCompletoRequest:
    """Valida uma linha (dict) contra ``LaudoCompletoRequest``.

    Usada pelo endpoint de upload CSV (§2.3): cada linha reaproveita o mesmo
    schema do laudo completo.
    """
    return LaudoCompletoRequest.model_validate(row)
