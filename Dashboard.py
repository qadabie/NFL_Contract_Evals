import pandas as pd

df = pd.read_csv("RB_survival_probs.csv")
#print(len(df))
df2 = pd.read_csv("processed_RB_data.csv")
#print(len(df2))

df = pd.merge(df,df2[['Player Name','Pick','Team']], left_on = 'Player Name',right_on="Player Name",how = 'left')

avg_rnd_surv = {}
for i in range(5):
    rnd = i+1
    pick = rnd*32
    df_rnd = df[(df['Pick_y']<=pick)&(df['Pick_y']>(pick-32))]
    surv_avg = dict(df_rnd[['Survival_Prob_1_seasons',  'Survival_Prob_2_seasons',
                 'Survival_Prob_3_seasons',  'Survival_Prob_4_seasons', 'Survival_Prob_5_seasons',
                 'Survival_Prob_6_seasons','Survival_Prob_7_seasons']].mean(axis=0))
    avg_rnd_surv[rnd] = {'career_probs':surv_avg}
    skills_avg = dict(df_rnd[['balance_score','security_score','protection_score','vision_score','yards_score']].mean(axis=0))
    avg_rnd_surv[rnd]['skills']= skills_avg
    
print(avg_rnd_surv[1]['career_probs'])
players = {}
for i in range(len(df)):
    player_dict = {}
    row = df.iloc[i]
    player_dict['Team']= row['Team']
    player_dict['Pick'] = row['Pick_y']
    player_dict['career_probs'] = dict(row[['Survival_Prob_1_seasons',  'Survival_Prob_2_seasons',
                 'Survival_Prob_3_seasons',  'Survival_Prob_4_seasons', 'Survival_Prob_5_seasons',
                 'Survival_Prob_6_seasons','Survival_Prob_7_seasons']])
    player_dict['skills'] = dict(row[['balance_score','security_score','protection_score','vision_score','yards_score']])
    name = str(row['Player Name'])
    #print(name)
    players[name] = player_dict

#print(df['Player Name'].head())
print(players['Alvin Kamara'])

import plotly.graph_objects as go
from plotly.subplots import make_subplots

team_colors = {
        "Seattle Seahawks": ("#69BE28", "#002244"),
        "San Francisco 49ers": ("#AA0000", "#B3995D"),
        "Los Angeles Rams": ("#003594", "#FFA300"),
        "Arizona Cardinals": ("#97233F", "#FFB612"),
        "Las Vegas Raiders": ("#A5ACAF", "#000000"),
        "Kansas City Chiefs": ("#E31837", "#FFB81C"),
        "Los Angeles Chargers": ("#0080C6", "#FFC20E"),
        "Denver Broncos": ("#FB4F14", "#002244"),
        "Chicago Bears": ("#C83803", "#0B162A"),
        "Green Bay Packers": ("#203731", "#FFB612"),
        "Detriot Lions": ("#0076B6", "#B0B7BC"),
        "Minnesota Vikings": ("#4F2683", "#FFC62F"),
        "Baltimore Ravens": ("#241773", "#9E7C0C"),
        "Cleveland Browns": ("#FF3C00", "#311D00"),
        "Pittsburgh Steelers": ("#FFB612", "#101820"),
        "Cincinnati Bengals": ("#FB4F14", "#000000"),
        "New England Patriots": ("#002244", "#C60C30"),
        "Buffalo Bills": ("#00338D", "#C60C30"),
        "Miami Dolphins": ("#008E97", "#FC4C02"),
        "New York Jets": ("#125740", "#FFFFFF"),
        "Dallas Cowboys": ("#003594", "#869397"),
        "New York Giants": ("#0B2265", "#A71930"),
        "Washington Commanders": ("#5A1414", "#FFB612"),
        "Philadelphia Eagles": ("#004C54", "#A5ACAF"),
        "Indianapolis Colts": ("#002C5F", "#A2AAAD"),
        "Houston Texans": ("#03202F", "#A71930"),
        "Tennessee Titans": ("#0C2340", "#4B92DB"),
        "Jacksonville Jaguars": ("#006778", "#D7A22A"),
        "Atlanta Falcons": ("#A71930", "#000000"),
        "Carolina Panthers": ("#0085CA", "#101820"),
        "New Orleans Saints": ("#D3BC8D", "#101820"),
        "Tampa Bay Buccaneers": ("#D50A0A", "#FF7900")
    }

def plot_player_dashboard(player_name):
    # NFL team colors (primary, secondary)
    player_data = players[player_name]
    team = player_data['Team']
    primary_color, secondary_color = team_colors.get(team, ("#333333", "#AAAAAA"))

    # Unpack player + average data
    skills = player_data["skills"]
    career_probs = player_data["career_probs"]
    #accolades = player_data["accolades"]
   
    pick = player_data['Pick']
    print(pick)
    rnd = (pick // 32)+1
    average_data = avg_rnd_surv[rnd]
    avg_skills = average_data["skills"]
    avg_career_probs = average_data["career_probs"]
    #avg_accolades = average_data["accolades"]

    # Radar chart prep
    radar_labels = list(skills.keys()) + [list(skills.keys())[0]]
    player_radar = list(skills.values()) + [list(skills.values())[0]]
    avg_radar = list(avg_skills.values()) + [list(avg_skills.values())[0]]

    # Line chart prep
    years = list(career_probs.keys())
    player_career = list(career_probs.values())
    avg_career = list(avg_career_probs.values())

    # Bar chart prep
    #accolade_labels = list(accolades.keys())
    #player_acc = list(accolades.values())
    #avg_acc = list(avg_accolades.values())

    # Setup figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Skill Radar", "Career Longevity"),
        specs=[[{"type": "polar"}, {"type": "xy"}]]
    )

    # Traces
    fig.add_trace(go.Scatterpolar(r=player_radar, theta=radar_labels, fill='toself', name='Player', line_color=primary_color), row=1, col=1)
    fig.add_trace(go.Scatterpolar(r=avg_radar, theta=radar_labels, fill='toself', name='Avg RB', line_color='gray', visible='legendonly'), row=1, col=1)

    fig.add_trace(go.Scatter(x=years, y=player_career, mode='lines+markers', name='Player', line=dict(color=primary_color, width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=years, y=avg_career, mode='lines+markers', name='Avg RB', line=dict(color='gray', dash='dash'), visible='legendonly'), row=1, col=2)

    #fig.add_trace(go.Bar(x=accolade_labels, y=player_acc, name='Player', marker_color=secondary_color), row=1, col=3)
    #fig.add_trace(go.Bar(x=accolade_labels, y=avg_acc, name='Avg RB', marker_color='gray', visible='legendonly'), row=1, col=3)

    # Buttons to toggle average overlays
    fig.update_layout(
        title_text=f"{player_name} – Career Projection Dashboard",
        title_font_size=20,
        height=500,
        showlegend=True,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5,
            y=-0.15,
            buttons=[
                dict(label="Player Only", method="update", args=[{"visible": [True, False, True, False, True, False]}]),
                dict(label="Player + Avg RB", method="update", args=[{"visible": [True, True, True, True, True, True]}])
            ]
        )]
    )

    fig.update_yaxes(range=[0, 1.05], row=1, col=2)
    fig.update_yaxes(range=[0, 1.05], row=1, col=3)

    fig.show()


plot_player_dashboard('TreVeyon Henderson')