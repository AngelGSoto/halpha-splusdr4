import dash
from dash import dcc, html, Input, Output, State
import dash_canvas
from dash_canvas.utils import parse_jsonstring
import numpy as np
import base64
import io
from PIL import Image

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dibuja líneas sobre tu imagen y calcula la pendiente"),
    dcc.Upload(
        id="upload-image",
        children=html.Div(["Arrastra y suelta o haz clic para subir un archivo"]),
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
        },
        accept="image/*",
    ),
    html.Div(id="image-container"),
    dash_canvas.DashCanvas(
        id="canvas",
        lineWidth=2,
        lineColor="blue",
        width=800,
        height=600,
    ),
    html.Button("Calcular", id="calculate-button", n_clicks=0),
    html.Div(id="results", style={"marginTop": "20px"}),
])

@app.callback(
    Output("canvas", "image_content"),
    [Input("upload-image", "contents")],
    prevent_initial_call=True
)
def update_canvas(image_content):
    if image_content is None:
        return None
    return image_content

@app.callback(
    Output("results", "children"),
    [Input("calculate-button", "n_clicks")],
    [State("canvas", "json_data")]
)
def calculate_slope(n_clicks, json_data):
    if n_clicks == 0 or json_data is None:
        return "Dibuja líneas y haz clic en calcular."
    
    try:
        # Parse the JSON data
        parsed_data = parse_jsonstring(json_data)
        print("Parsed JSON data:", parsed_data)  # Debugging line
        paths = parsed_data.get("objects", [])
    except (TypeError, KeyError, AttributeError) as e:
        print("Error al analizar los datos del canvas:", e)  # Debugging line
        return "Error al analizar los datos del canvas."

    results = []
    for path in paths:
        print("Path data:", path)  # Debugging line
        if path["type"] == "line":
            x1, y1 = path["x1"], path["y1"]
            x2, y2 = path["x2"], path["y2"]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                results.append(f"Pendiente: {slope:.2f}, Intersección: {intercept:.2f}")
            else:
                results.append("Línea vertical: Pendiente infinita, Intersección N/A")
    
    if not results:
        return "No se encontraron líneas. Dibuja líneas en la imagen."
    return html.Ul([html.Li(result) for result in results])

if __name__ == "__main__":
    app.run_server(debug=True)
