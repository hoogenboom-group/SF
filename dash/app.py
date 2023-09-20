from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import pandas as pd

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H4("Re-scaling factor"),
        html.Div([
            html.Div([
                html.P("Adjust sliders to set plot parameters"),
                html.Div([ html.Label("Numerical aperture"),
                           dcc.Slider(min=0.1, max=1.5, step=0.05, marks={it:f"{it:.2f}" for it in np.arange(0.1,1.7,0.2)}, value=0.85, id='na-slider')
                         ], className="rows"),
                html.Div([ html.Br(),
                           html.Label("Immersion refractive index (n1)"),
                           dcc.Slider(min=1, max=1.52, step=0.01, marks={it:f"{it:.2f}" for it in np.arange(1,1.52,0.1)}, value=1, id='n1-slider'),
                         ], className="rows"),
                html.Div([ html.Br(),
                           html.Label("Sample refractive index (n2)"),
                           dcc.Slider(min=1, max=1.52, step=0.01, marks={it:f"{it:.2f}" for it in np.arange(1,1.52,0.1)}, value=1.33, id='n2-slider'),
                         ], className="rows"),
                html.Div([ html.Br(),
                           html.Label("Wavelength [um]"),
                           dcc.Slider(min=0.2, max=0.9, step=0.01, marks={it:f"{it:.2f}" for it in np.arange(0.2,0.95,0.05)}, value=0.51, id='lambda-slider')
                         ], className="rows"),
                html.Div([ html.Br(),
                           html.Div([
                               html.Label("Show:"),
                               dcc.Checklist(options=[
                                           {'label': 'All', 'value': 'ALL'},
                                           {'label': 'Focal shift', 'value': 'FS'},
                                           {'label': 'Carlsson', 'value': 'C'},
                                           {'label': 'Visser', 'value': 'V'},
                                           {'label': 'Diel (median)', 'value': 'DMED'},
                                           {'label': 'Diel (mean)', 'value': 'DMEA'},
                                           {'label': 'Stallinga', 'value': 'S'},
                                           {'label': 'Lyakin', 'value': 'L'},
                                          ],
                                          value=['MTL'], id='checklist'
                               ),
                           ], className="four columns"),
                           html.Div([
                            html.Br(),
                            html.Button("Download CSV data", id="btn_csv"),
                            dcc.Download(id="download-dataframe-csv"),
                            html.Div([html.Br(),
                            html.A(html.Button('Reset plot'),href='/')]),
                           ], className="four columns"),
                         ], className="rows"),
            ], className="five columns"),
            html.Div([
                dcc.Graph(id="graph",style={'height': '90vh'}, config = {'displayModeBar': True}),
            ], className="seven columns"),
        ], className="row")
    ]
)


def scaling_factor(NA,n1,n2,lam_0):
    z = np.append(np.arange(lam_0,10,0.1), [np.arange(10,100,1), np.arange(100,1000,10), np.arange(1000,10000,100)]) # np.arange(0.1,10,0.1)
    n2overn1 = np.divide(n2,n1)

    if n2overn1 < 1: eps = np.multiply(-1,np.divide(np.divide(lam_0,4),(np.multiply(z,n2))))
    else: eps = np.divide(np.divide(lam_0,4),(np.multiply(z,n2)))
    eps_term = np.multiply(eps, np.subtract(2,eps))

    m = np.emath.sqrt(np.subtract(np.power(n2,2),np.power(n1,2)))

    sf_univ = np.multiply(np.divide(n2,n1),
                          np.divide(1-eps+np.divide(m,n1)*np.emath.sqrt(eps_term),
                                    1-np.multiply(np.divide(n2,n1)**2,eps_term)))
    sf_crit = np.divide(n1-np.emath.sqrt(np.power(n1,2)-np.power(NA,2)),
                        n2-np.emath.sqrt(np.power(n2,2)-np.power(NA,2)))

    sf = np.zeros(len(z))
    for i in range(len(sf)):
        if n2overn1 < 1: sf[i] = np.max([np.real(sf_univ[i]),np.real(sf_crit)])
        elif n2overn1 > 1:sf[i] = np.min([np.real(sf_univ[i]),np.real(sf_crit)])
        else: sf[i]=1
    return z,sf,sf_crit,n2overn1

def Carlsson(z,n2overn1):
    return np.zeros(len(z)) + n2overn1

def Lyakin(z,n_sample,n_im,NA):
    d = 1
    top = np.add(n_im,np.sqrt(np.subtract(np.power(n_im,2),np.power(NA,2))))
    bottom_1 = np.multiply(4,np.subtract(np.power(n_sample,2),np.power(n_im,2)))
    bottom_2 = np.add(n_im,np.emath.sqrt(np.subtract(np.power(n_im,2),np.power(NA,2))))
    bottom = np.real(np.emath.sqrt(np.add(bottom_1,np.power(bottom_2,2))))
    if bottom == 0: bottom=0.000000000000001
    dz = np.multiply(d,np.divide(top,bottom))
    scaling_factor = np.divide(1,dz)
    return np.zeros(len(z)) + scaling_factor

def visser(z,n_sample,n_im,NA):
    top = np.tan(np.arcsin(np.divide(NA,n_im)))
    bottom = np.tan(np.arcsin(np.divide(NA,n_sample)))
    sf = np.divide(top,bottom)
    return np.zeros(len(z)) + sf

def diel_mean(z,n_im,n_sample,NA):
    sum=0
    number_of_rays=10000 # paper uses 100, but this is still doable.
    for i in range(number_of_rays):
        k=i+1
        top     =  np.tan(np.arcsin(np.divide((NA*k),(np.multiply(number_of_rays,n_im)))))
        bottom  =  np.tan(np.arcsin(np.divide((NA*k),(np.multiply(number_of_rays,n_sample)))))
        sum +=np.divide(top,bottom)
    return np.zeros(len(z)) + np.divide(sum,number_of_rays)

def diel_median(z,n_im,n_sample,NA):
    top = np.tan(np.arcsin(np.divide(0.5*NA,n_im)))
    bottom = np.tan(np.arcsin(np.divide(0.5*NA,n_sample)))
    return np.zeros(len(z)) + np.divide(top,bottom)

def stallinga_high(z,nn1,nn2,NA):
    if nn1==nn2:
        return np.ones(len(z))
    alphas=[]
    for i in range(len(NA)):
        alphas.append(-1*(f1f2_av(nn1,nn2,NA[i])
                          - f_av(nn1,NA[i]) * f_av(nn2,NA[i]))
                      / (ff_av(nn1,NA[i]) - f_av(nn1,NA[i])**2))
    d=1
    dz = np.multiply((np.add(alphas,1)),d) #we take delta d as 1
    scaling_factor = np.divide((d-dz),d)

    sf = np.divide(-1,alphas)
    return np.zeros(len(z)) + sf

def f_av(nn,NA):
    f=2*(nn**3-(nn**2-NA**2)**(3/2))/(3*NA**2)
    return f

def ff_av(nn,NA):
    ff=nn**2-(NA**2)/2
    return ff

def f1f2_av(nn1,nn2,NA):
    f1f2 = ( (nn1*nn2**3+nn2*nn1**3 - (nn1**2+nn2**2-2*NA**2)*np.sqrt(nn1**2-NA**2)*np.sqrt(nn2**2-NA**2)
            - ((nn1**2-nn2**2)**2)*np.log( ( np.sqrt(nn1**2-NA**2) - np.sqrt(nn2**2-NA**2) )/ (nn1-nn2) ) )/(4*NA**2) )
    return f1f2

def sergey_analytical_sf(NA, n1, n2, lambda0):
    z_a = np.append(np.arange(0.5, 100, 0.1), 1000000)
    if n2 > n1:
        m = np.sqrt(n2**2 - n1**2)
    else:
        m = np.sqrt(n1**2 - n2**2)
    epsilon = (lambda0/4) / (z_a * n2)
    if n2 > n1:
        xi_univ = (n2/n1) * ((1-epsilon) + (m/n1**2) * np.sqrt(2*epsilon - epsilon**2)) / (1-(n2/n1)**2 *epsilon * (2-epsilon))
        xi_crit = (n1 - np.sqrt(n1**2 - NA**2)) / (n2 - np.sqrt(n2**2 - NA**2))
        condi = np.where(xi_univ < xi_crit)
        if condi[0].size > 0:
            idx = condi[0][0]
            xi_univ[:idx] = xi_crit
    else:
        xi_univ = (n2/n1) * 1/(((1-epsilon) + (m/n1**2) * np.sqrt(2*epsilon - epsilon**2)) / (1-(n2/n1)**2 *epsilon * (2-epsilon)))
    f = interp1d(z_a, xi_univ, fill_value="extrapolate", kind="slinear")
    z_a = np.arange(lambda0, 100, 0.1)
    return z_a, f(z_a)

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn_csv", "n_clicks"),
    State('lambda-slider', 'value'),
    State('n1-slider', 'value'),
    State('n2-slider', 'value'),
    State('na-slider', 'value'),
    State('checklist', 'value')],
    prevent_initial_call=True,
    )
def func(n_clicks, lamb, n1, n2, NA, chks):
    fig, _ = display_graph(lamb, n1, n2, NA, chks)
    df = pd.DataFrame({d.name:pd.Series(d.y, index=d.x) for d in fig['data']})
    df.index.name = 'Depth [um]'
    return dcc.send_data_frame(df.to_csv, "Re-scaling factor.csv")

@app.callback(
    [Output("graph", "figure"),
     Output("na-slider", "max"),
    ],
    [Input('lambda-slider', 'value'),
    Input('n1-slider', 'value'),
    Input('n2-slider', 'value'),
    Input('na-slider', 'value'),
    Input('checklist', 'value')])
def display_graph(lamb, n1, n2, NA, chks):
    if "ALL" in chks: chks = ["FS", "C", "L", "DMED", "DMEA", "S", "V"]
    z, sf, sf_crit, n2overn1 = scaling_factor(NA, n1, n2, lamb)
    z = np.insert(z, 0, [0])
    sf = np.insert(sf, 0, [sf[0]])
    ymi, yma = min(sf), max(sf)
    fs = z*(1/sf-1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Line(x=z, y=sf, name="Axial re-scaling factor"),
        secondary_y=False,
    )
    if "FS" in chks:
        fig.add_trace(
            go.Line(x=z, y=fs, name="Focal shift [um]", line_color = 'black'),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=z, y=z*(1/Lyakin(z,n2,n1,NA)-1), name='Focal shift (Lyakin) [um]', line = dict(color='black', dash='dash')),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=z, y=z*(1/diel_median(z,n1,n2,NA)-1), name='Focal shift (Diel, median) [um]', line = dict(color='black', dash='dot')),
            secondary_y=True,
        )
    
    yma2 = fs[190]
    
    if "S" in chks and n2overn1 >= 1:
        y = stallinga_high(z,n1,n2,[NA])
        fig.add_trace(
            go.Line(x=z, y=y, name="Stallinga (2005)"),
            secondary_y=False,
        )
        ymi = min([ymi, y[0]])
        yma = max([yma, y[0]])
    if "C" in chks:
        y = Carlsson(z,n2overn1)
        fig.add_trace(
            go.Line(x=z, y=y, name="Carlsson (1991)"),
            secondary_y=False,
        )
        ymi = min([ymi, y[0]])
        yma = max([yma, y[0]])
    if "V" in chks and n2 > NA:
        y = visser(z,n2,n1,NA)
        fig.add_trace(
            go.Line(x=z, y=y, name="Visser & Oud (1992)"),
            secondary_y=False,
        )
        ymi = min([ymi, y[0]])
        yma = max([yma, y[0]])
    if "L" in chks:
        y = Lyakin(z,n2,n1,NA)
        fig.add_trace(
            go.Line(x=z, y=y, name="Lyakin et al (2017)"),
            secondary_y=False,
        )
        ymi = min([ymi, y[0]])
        yma = max([yma, y[0]])
    if "DMED" in chks:
        y = diel_median(z,n1,n2,NA)
        fig.add_trace(
            go.Line(x=z, y=y, name="Diel et al (median, 2020)"),
            secondary_y=False,
        )
        ymi = min([ymi, y[0]])
        yma = max([yma, y[0]])
    if "DMEA" in chks and n2 > NA:
        y = diel_mean(z,n1,n2,NA)
        fig.add_trace(
            go.Line(x=z, y=y, name="Diel et al (means, 2020)"),
            secondary_y=False,
        )
        ymi = min([ymi, y[0]])
        yma = max([yma, y[0]])
    xmi, xma = 0,100
    fig.update_xaxes(title_text="<b>Depth [um]</b>", range=[xmi, xma])
    fig.update_yaxes(title_text="<b>Axial re-scaling factor</b>", range=[ymi-0.1, yma+0.1], secondary_y=False)
    fig.update_yaxes(title_text="<b>Focal shift [um]</b>", secondary_y=True, range=[yma2, 1])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.25,
                            x=0,
                            xanchor="left",
                    )
    )
    return fig, n1

if __name__ == "__main__":
    app.run_server(debug=True) #