from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class InspecaoMedidorRequest(BaseModel):
    id_medidor: int = Field(..., description="ID do medidor a ser inspecionado")
    data_inspecao: str = Field(..., description="Data da inspeção do medidor")
    
    class Config:
        json_schema_extra = {
            "example":{
                "id_medidor": 123,
                "data_inspecao": "2024-06-01"
            }
        }
