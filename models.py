from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List

class DiagramInput(BaseModel):
    layer_sizes: List[int] = Field(
        ...,
        title="Layer Sizes",
        description="Number of nodes in each layer. Example: [4, 5, 8, 3]"
    )
    colors: List[str] = Field(
        ...,
        title="Colors",
        description="List of colors for each layer. Should match the number of layers."
    )
    bias_color: str = Field(
        ...,
        title="Bias Color",
        description="Color to use for the bias nodes."
    )

    @model_validator(mode="after")
    def validate_colors_match_layers(self):
        if len(self.layer_sizes) != len(self.colors):
            raise ValueError("The number of colors must match the number of layers.")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "layer_sizes": [4, 5, 8, 3],
                "colors": ["red", "blue", "green", "purple"],
                "bias_color": "gray"
            }
        }


class DiagramOutput(BaseModel):
    svg: str = Field(
        ...,
        title="SVG Output",
        description="The generated SVG as a string."
    )
    node_count: int = Field(
        ...,
        title="Total Nodes",
        description="Total number of nodes in all layers (excluding bias)."
    )
    layer_count: int = Field(
        ...,
        title="Total Layers",
        description="Total number of layers."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "svg": "<svg width='500' height='300'>...</svg>",
                "node_count": 20,
                "layer_count": 4
            }
        }