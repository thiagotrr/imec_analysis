from pydantic import BaseModel, Field

class InspecaoMedidorResponse(BaseModel):
    id_medidor: int = Field(..., description="ID do medidor inspecionado.")
    data_inspecao: str = Field(..., description="Data da inspeção do medidor.")
    resultado: str = Field(..., description="Resultado da inspeção do medidor, conforme treinamento do modelo.")
    resultado_detalhado: str = Field(..., description="Resultado da inspeção do medidor, com considerações acerca dos dados enviados inferência do modelo.")
    
    class Config:
        json_schema_extra = {
            "example":{
                "id_medidor": 123,
                "data_inspecao": "2024-06-01",
                "resultado": "Aprovado"
            }
        }