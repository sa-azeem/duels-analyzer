import streamlit as st
import random
from math import floor 
import pandas as pd, numpy as np, requests, json
import datetime
# import helpers
import plotly.express as px
import altair as alt

class helpers:
    @staticmethod
    def get_session(ncfa):
            session = requests.Session()
            session.cookies.set("_ncfa", ncfa, domain="www.geoguessr.com")
            return session
    @staticmethod
    def get_duel_tokens(session):
        BASE_URL_V4 = "https://www.geoguessr.com/api/v4"
        # only get competitive duels tokens
        game_tokens = []
        pagination_token = None
        while True:
            response = session.get(f"{BASE_URL_V4}/feed/private", params={'paginationToken': pagination_token})
            pagination_token = entries = response.json()['paginationToken']
            entries = response.json()['entries']
            for entry in entries:
                payload_json = json.loads(entry['payload'])
                if type(payload_json) is dict:
                    try:
                        if payload_json['gameMode'] == 'Duels' and 'competitiveGameMode' in payload_json:
                            game_tokens.append(payload_json['gameId'])
                    except Exception as e:
                        continue
                else:
                    for payload in payload_json:
                        try:
                            if payload['payload']['gameMode'] == 'Duels' and 'competitiveGameMode' in payload['payload']:
                                game_tokens.append(payload['payload']['gameId'])
                        except Exception as e:
                            continue
            if not pagination_token:
                break
        return game_tokens
    @staticmethod
    def get_player_data(session):
        BASE_URL_V4 = "https://www.geoguessr.com/api/v4"
        try:
            player_data=session.get(f"{BASE_URL_V4}/feed/private").json()['entries'][0]['user']
        except:
            return {}
        return {'id':player_data['id'],
        'nick':player_data['nick']}
    @staticmethod
    def get_duels(session,duel_tokens,my_player_Id,loading_bar):
        data_dict=dict({'Date':[],
                    'Game Id':[],
                    'Round Number':[],
                    'Country':[],
                    'Latitude':[],
                    'Longitude':[],
                    'Damage Multiplier':[],
                    'Opponent Id':[],
                    'Opponent Country':[],
                    'Your Latitude':[],
                    'Your Longitude':[],
                    'Opponent Latitude':[],
                    'Opponent Longitude':[],
                    'Your Distance':[],
                    'Opponent Distance':[],
                    'Your Score':[],
                    'Opponent Score':[],
                    'Map Name':[],
                    'Game Mode':[],
                    'Moving':[],
                    'Zooming':[],
                    'Rotating':[],
                    'Your Rating':[],
                    'Opponent Rating':[],
                    'Score Difference':[],
                    'Win Percentage':[]
                    })

        BASE_URL_V3="https://game-server.geoguessr.com/api/duels"
        count_=0
        for token in duel_tokens:
            count_+=1
            loading_bar.progress(count_/len(duel_tokens))
            response = session.get(f"{BASE_URL_V3}/{token}")
            if response.status_code == 200:
                game = response.json()
                me=0
                other=1
                if game['teams'][1]['players'][0]['playerId']==my_player_Id:
                    me=1
                    other=0

                for i in range (game['currentRoundNumber']):
                    round=game['rounds'][i]

                    data_dict['Round Number'].append(round['roundNumber'])
                    data_dict['Country'].append(helpers.get_country_name(round['panorama']['countryCode']))
                    data_dict['Latitude'].append(round['panorama']['lat'])
                    data_dict['Longitude'].append(round['panorama']['lng'])
                    data_dict['Damage Multiplier'].append(round['damageMultiplier'])

                    # if no guess is made, there is no entry in guesses of that round, so we find if the round number in round and guess are same, if not, then NAN.
                    my_guess=  [guess for guess in game['teams'][me]['players'][0]['guesses'] if guess['roundNumber']==i+1]
                    if my_guess:
                        my_guess=my_guess[0]
                        data_dict['Your Latitude'].append(my_guess['lat'])
                        data_dict['Your Longitude'].append(my_guess['lng'])
                        data_dict['Your Distance'].append(my_guess['distance']/1000)
                        data_dict['Your Score'].append(my_guess['score'])
                    else:
                        data_dict['Your Latitude'].append(0)
                        data_dict['Your Longitude'].append(0)
                        data_dict['Your Distance'].append(0)
                        data_dict['Your Score'].append(0)


                    other_guess=  [guess for guess in game['teams'][other]['players'][0]['guesses'] if guess['roundNumber']==i+1]
                    if other_guess:
                        other_guess=other_guess[0]
                        data_dict['Opponent Latitude'].append(other_guess['lat'])
                        data_dict['Opponent Longitude'].append(other_guess['lng'])
                        data_dict['Opponent Distance'].append(other_guess['distance']/1000)
                        data_dict['Opponent Score'].append(other_guess['score'])
                    else:
                        data_dict['Opponent Latitude'].append(0)
                        data_dict['Opponent Longitude'].append(0)
                        data_dict['Opponent Distance'].append(0)
                        data_dict['Opponent Score'].append(0)
                    data_dict['Score Difference'].append(
                        data_dict['Your Score'][-1]-data_dict['Opponent Score'][-1]
                    )
                    data_dict['Win Percentage'].append(
                        int(data_dict['Your Score'][-1]>data_dict['Opponent Score'][-1])*100
                    )
                    # repeated
                    data_dict['Game Id'].append(game['gameId'])

                    data_dict['Date'].append(game['rounds'][0]['startTime'])

                    data_dict['Map Name'].append(game['options']['map']['name'])
                    data_dict['Game Mode'].append( game['options']['competitiveGameMode'])

                    data_dict['Moving'].append(not game['options']['movementOptions']['forbidMoving'])
                    data_dict['Zooming'].append(not game['options']['movementOptions']['forbidZooming'])
                    data_dict['Rotating'].append(not game['options']['movementOptions']['forbidRotating'])

                    data_dict['Opponent Id'].append(game['teams'][other]['players'][0]['playerId'])
                    data_dict['Opponent Country'].append( helpers.get_country_name(game['teams'][other]['players'][0]['countryCode']))

                    data_dict['Your Rating'].append(game['teams'][me]['players'][0]['rating'])
                    data_dict['Opponent Rating'].append(game['teams'][other]['players'][0]['rating'])

            else:
                # print(f"Request failed with status code: {response.status_code}")
                # print(f"Response content: {response.text}")
                pass 
        return data_dict
    @staticmethod
    def datetime_processing(df):
        from datetime import timedelta
        def utc_to_offset(series):
            return series + timedelta(hours=5, minutes=30)
        df['Date']=pd.to_datetime(df['Date'],format="%Y-%m-%dT%H:%M:%S.%f%z",errors='coerce').fillna(pd.to_datetime(df['Date'],format="%Y-%m-%dT%H:%M:%S%z",errors='coerce'))
        df['Date']=utc_to_offset(df['Date'])
        df['Time']=df['Date'].dt.time
        df['Date']=df['Date'].dt.date
        df['Hour']=df['Time'].apply(lambda x: x.hour)
        return df
    @staticmethod
    def groupby_country(df):
        by_country=df.groupby('Country').agg({'Your Score': 'mean', 'Opponent Score': 'mean','Score Difference':'mean','Win Percentage':'mean','Country':'count','Your Distance':'mean'})
        by_country.rename(columns={'Country':'Number of Rounds','Your Distance':'Distance'}, inplace=True)
        # by_country['Score Difference']=by_country['Your Score']-by_country['Opponent Score']
        by_country['Win Percentage']=by_country['Win Percentage'].apply(lambda x: round(x,2))
        by_country[['Your Score', 'Opponent Score', 'Score Difference','Distance']]=by_country[['Your Score', 'Opponent Score', 'Score Difference','Distance']].apply(round)
        

        new_cols=['Number of Rounds']+[col for col in by_country.columns if col != 'Number of Rounds']
        by_country=by_country[new_cols]
        return by_country
    @staticmethod
    def get_country_name(country_code): 
        country_code=country_code.lower()
        country_name_dict={'ad': 'Andorra',
                'ae': 'United Arab Emirates',
                'af': 'Afghanistan',
                'ag': 'Antigua and Barbuda',
                'ai': 'Anguilla',
                'al': 'Albania',
                'am': 'Armenia',
                'ao': 'Angola',
                'aq': 'Antarctica',
                'ar': 'Argentina',
                'as': 'American Samoa',
                'at': 'Austria',
                'au': 'Australia',
                'aw': 'Aruba',
                'ax': 'Åland Islands',
                'az': 'Azerbaijan',
                'ba': 'Bosnia and Herzegovina',
                'bb': 'Barbados',
                'bd': 'Bangladesh',
                'be': 'Belgium',
                'bf': 'Burkina Faso',
                'bg': 'Bulgaria',
                'bh': 'Bahrain',
                'bi': 'Burundi',
                'bj': 'Benin',
                'bl': 'Saint Barthélemy',
                'bm': 'Bermuda',
                'bn': 'Brunei Darussalam',
                'bo': 'Bolivia',
                'bq': 'Bonaire, Sint Eustatius and Saba',
                'br': 'Brazil',
                'bs': 'Bahamas',
                'bt': 'Bhutan',
                'bv': 'Bouvet Island',
                'bw': 'Botswana',
                'by': 'Belarus',
                'bz': 'Belize',
                'ca': 'Canada',
                'cc': 'Cocos (Keeling) Islands',
                'cd': 'Congo (Democratic Republic of the)',
                'cf': 'Central African Republic',
                'cg': 'Congo',
                'ch': 'Switzerland',
                'ci': 'Côte d\'Ivoire',
                'ck': 'Cook Islands',
                'cl': 'Chile',
                'cm': 'Cameroon',
                'cn': 'China',
                'co': 'Colombia',
                'cr': 'Costa Rica',
                'cu': 'Cuba',
                'cv': 'Cabo Verde',
                'cw': 'Curaçao',
                'cx': 'Christmas Island',
                'cy': 'Cyprus',
                'cz': 'Czechia',
                'de': 'Germany',
                'dj': 'Djibouti',
                'dk': 'Denmark',
                'dm': 'Dominica',
                'do': 'Dominican Republic',
                'dz': 'Algeria',
                'ec': 'Ecuador',
                'ee': 'Estonia',
                'eg': 'Egypt',
                'eh': 'Western Sahara',
                'er': 'Eritrea',
                'es': 'Spain',
                'et': 'Ethiopia',
                'fi': 'Finland',
                'fj': 'Fiji',
                'fk': 'Falkland Islands (Malvinas)',
                'fm': 'Micronesia (Federated States of)',
                'fo': 'Faroe Islands',
                'fr': 'France',
                'ga': 'Gabon',
                'gb': 'United Kingdom',
                'gd': 'Grenada',
                'ge': 'Georgia',
                'gf': 'French Guiana',
                'gg': 'Guernsey',
                'gh': 'Ghana',
                'gi': 'Gibraltar',
                'gl': 'Greenland',
                'gm': 'Gambia',
                'gn': 'Guinea',
                'gp': 'Guadeloupe',
                'gq': 'Equatorial Guinea',
                'gr': 'Greece',
                'gs': 'South Georgia and the South Sandwich Islands',
                'gt': 'Guatemala',
                'gu': 'Guam',
                'gw': 'Guinea-Bissau',
                'gy': 'Guyana',
                'hk': 'Hong Kong',
                'hm': 'Heard Island and McDonald Islands',
                'hn': 'Honduras',
                'hr': 'Croatia',
                'ht': 'Haiti',
                'hu': 'Hungary',
                'id': 'Indonesia',
                'ie': 'Ireland',
                'il': 'Israel',
                'im': 'Isle of Man',
                'in': 'India',
                'io': 'British Indian Ocean Territory',
                'iq': 'Iraq',
                'ir': 'Iran',
                'is': 'Iceland',
                'it': 'Italy',
                'je': 'Jersey',
                'jm': 'Jamaica',
                'jo': 'Jordan',
                'jp': 'Japan',
                'ke': 'Kenya',
                'kg': 'Kyrgyzstan',
                'kh': 'Cambodia',
                'ki': 'Kiribati',
                'km': 'Comoros',
                'kn': 'Saint Kitts and Nevis',
                'kp': 'North Korea',
                'kr': 'South Korea',
                'kw': 'Kuwait',
                'ky': 'Cayman Islands',
                'kz': 'Kazakhstan',
                'la': 'Laos',
                'lb': 'Lebanon',
                'lc': 'Saint Lucia',
                'li': 'Liechtenstein',
                'lk': 'Sri Lanka',
                'lr': 'Liberia',
                'ls': 'Lesotho',
                'lt': 'Lithuania',
                'lu': 'Luxembourg',
                'lv': 'Latvia',
                'ly': 'Libya',
                'ma': 'Morocco',
                'mc': 'Monaco',
                'md': 'Moldova',
                'me': 'Montenegro',
                'mf': 'Saint Martin',
                'mg': 'Madagascar',
                'mh': 'Marshall Islands',
                'mk': 'North Macedonia',
                'ml': 'Mali',
                'mm': 'Myanmar',
                'mn': 'Mongolia',
                'mo': 'Macao',
                'mp': 'Northern Mariana Islands',
                'mq': 'Martinique',
                'mr': 'Mauritania',
                'ms': 'Montserrat',
                'mt': 'Malta',
                'mu': 'Mauritius',
                'mv': 'Maldives',
                'mw': 'Malawi',
                'mx': 'Mexico',
                'my': 'Malaysia',
                'mz': 'Mozambique',
                'na': 'Namibia',
                'nc': 'New Caledonia',
                'ne': 'Niger',
                'nf': 'Norfolk Island',
                'ng': 'Nigeria',
                'ni': 'Nicaragua',
                'nl': 'Netherlands',
                'no': 'Norway',
                'np': 'Nepal',
                'nr': 'Nauru',
                'nu': 'Niue',
                'nz': 'New Zealand',
                'om': 'Oman',
                'pa': 'Panama',
                'pe': 'Peru',
                'pf': 'French Polynesia',
                'pg': 'Papua New Guinea',
                'ph': 'Philippines',
                'pk': 'Pakistan',
                'pl': 'Poland',
                'pm': 'Saint Pierre and Miquelon',
                'pn': 'Pitcairn',
                'pr': 'Puerto Rico',
                'ps': 'Palestine',
                'pt': 'Portugal',
                'pw': 'Palau',
                'py': 'Paraguay',
                'qa': 'Qatar',
                're': 'Réunion',
                'ro': 'Romania',
                'rs': 'Serbia',
                'ru': 'Russia',
                'rw': 'Rwanda',
                'sa': 'Saudi Arabia',
                'sb': 'Solomon Islands',
                'sc': 'Seychelles',
                'sd': 'Sudan',
                'se': 'Sweden',
                'sg': 'Singapore',
                'sh': 'Saint Helena',
                'si': 'Slovenia',
                'sj': 'Svalbard and Jan Mayen',
                'sk': 'Slovakia',
                'sl': 'Sierra Leone',
                'sm': 'San Marino',
                'sn': 'Senegal',
                'so': 'Somalia',
                'sr': 'Suriname',
                'ss': 'South Sudan',
                'st': 'Sao Tome and Principe',
                'sv': 'El Salvador',
                'sx': 'Sint Maarten',
                'sy': 'Syria',
                'sz': 'Eswatini',
                'tc': 'Turks and Caicos Islands',
                'td': 'Chad',
                'tf': 'French Southern Territories',
                'tg': 'Togo',
                'th': 'Thailand',
                'tj': 'Tajikistan',
                'tk': 'Tokelau',
                'tl': 'Timor-Leste',
                'tm': 'Turkmenistan',
                'tn': 'Tunisia',
                'to': 'Tonga',
                'tr': 'Turkey',
                'tt': 'Trinidad and Tobago',
                'tv': 'Tuvalu',
                'tw': 'Taiwan',
                'tz': 'Tanzania',
                'ua': 'Ukraine',
                'ug': 'Uganda',
                'um': 'United States Minor Outlying Islands',
                'us': 'United States',
                'uy': 'Uruguay',
                'uz': 'Uzbekistan',
                'va': 'Vatican City',
                'vc': 'Saint Vincent and the Grenadines',
                've': 'Venezuela',
                'vg': 'British Virgin Islands',
                'vi': 'U.S. Virgin Islands',
                'vn': 'Vietnam',
                'vu': 'Vanuatu',
                'wf': 'Wallis and Futuna',
                'ws': 'Samoa',
                'xk': 'Kosovo',
                'ye': 'Yemen',
                'yt': 'Mayotte',
                'za': 'South Africa',
                'zm': 'Zambia',
                'zw': 'Zimbabwe',}
        if country_code in country_name_dict.keys():
            return country_name_dict[country_code]
        else:
            return country_code
    @staticmethod
    def display_country_scores_map(df, country_col, score_col):
        # reversing color is needed for distance because more is less in case of distance
        color_=px.colors.sequential.Turbo_r
        if score_col=='Distance':
            color_=px.colors.sequential.Turbo
        fig = px.choropleth(
            df,
            locations=country_col,
            locationmode="country names",
            color=score_col,
            hover_name=country_col,
            color_continuous_scale=color_,
        )
        st.plotly_chart(fig)
    @staticmethod
    def alt_chart(data,x,y):
        c = alt.Chart(data).mark_bar().encode(x=alt.X(x,sort=None), y=y)
        st.altair_chart(c)
    @staticmethod        
    def sorted_bar_chart(data,x,y,color_=False):
        col1, col2 = st.columns(2)
        with col1:
            checkbox1 = st.checkbox("Sort",key='1'+x+y)
        with col2:
            checkbox2 = st.checkbox("Descending",key='3'+x+y)
        
        data=data.reset_index()
        data[x]=data[x].astype('object')
        sorted_data=data.sort_values(by=y if checkbox1 else x,ascending=not checkbox2)
        c = alt.Chart(sorted_data).mark_bar().encode(x=alt.X(x,sort=None,type='nominal'),y=y)
        if color_:
            c = alt.Chart(sorted_data).mark_bar().encode(x=alt.X(x,sort=None,type='nominal'),y=y,color=color_)
        st.altair_chart(c)
    @staticmethod
    def groupby_round(df):
        by_round=df.groupby('Round Number').agg({'Your Score': 'mean', 'Opponent Score': 'mean','Round Number':'count','Your Distance':'mean'})
        by_round.rename(columns={'Round Number':'Number of Rounds','Your Distance':'Distance'}, inplace=True)
        by_round['Score Difference']=by_round['Your Score']-by_round['Opponent Score']
        by_round['Win Percentage']=df.groupby('Round Number')[['Your Score','Opponent Score']].apply(lambda x: (x['Your Score']>x['Opponent Score']).mean()*100).apply(lambda x: round(x,2))
        by_round[['Your Score', 'Opponent Score', 'Score Difference','Distance']]=by_round[['Your Score', 'Opponent Score', 'Score Difference','Distance']].apply(round)
        # new_cols=['Number of Rounds']+[col for col in by_round.columns if col != 'Number of Rounds']
        # by_round=by_round[new_cols]
        return by_round
    @staticmethod
    def groupby_date(df,date_options):
        st.write(df)
        by_date=df.groupby('Round Number').agg({'Your Score': 'mean', 'Opponent Score': 'mean','Round Number':'count','Your Distance':'mean'})
        by_date.rename(columns={'Round Number':'Number of Rounds','Your Distance':'Distance'}, inplace=True)
        by_date['Score Difference']=by_date['Your Score']-by_date['Opponent Score']
        # by_date['Win Percentage']=df.groupby('Country')[['Your Score','Opponent Score']].apply(lambda x: (x['Your Score']>x['Opponent Score']).mean()*100).apply(lambda x: round(x,2))
        by_date[['Your Score', 'Opponent Score', 'Score Difference','Distance']]=by_date[['Your Score', 'Opponent Score', 'Score Difference','Distance']].apply(round)
        # new_cols=['Number of Rounds']+[col for col in by_date.columns if col != 'Number of Rounds']
        # by_date=by_date[new_cols]
        return by_date
    @staticmethod
    def create_binned_histogram(df,  metric_col):
        date_col='Date'
        if metric_col=='Distance':
            metric_col='Your Distance'
        elif metric_col=='Score Difference':
            df['Score Difference']=df['Your Score']-df['Opponent Score']
        df['Win Percentage']=(df['Your Score']>df['Opponent Score']).apply(lambda x:int(x)*100)
        df[date_col]=pd.to_datetime(df[date_col])
        date_option = st.radio(
            "Bin by:",
            ("Week", "Month", "Year"),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        df['Date'] = pd.to_datetime(df['Date'])

        if date_option == 'Week':
            df['Group'] = df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
        elif date_option == 'Month':
            df['Group'] = df['Date'].dt.to_period('M').apply(lambda r: r.start_time)
        else:  # Yearly
            df['Group'] = df['Date'].dt.to_period('Y').apply(lambda r: r.start_time)
        
        fig = px.histogram(df, x='Group', y=metric_col, nbins=len(df['Group'].unique()), labels={'Group': 'Date'},histfunc='avg')

        fig.update_layout(bargap=0.1, xaxis_title=date_option, yaxis_title=metric_col)

        st.plotly_chart(fig, use_container_width=True)
    @staticmethod
    def groupby_country_against(df):
        by_country_against=df.groupby('Opponent Country').agg({'Your Score': 'mean', 'Opponent Score': 'mean','Country':'count','Your Distance':'mean'})
        by_country_against.rename(columns={'Country':'Number of Rounds','Your Distance':'Distance'}, inplace=True)
        by_country_against['Score Difference']=by_country_against['Your Score']-by_country_against['Opponent Score']
        by_country_against['Win Percentage']=df.groupby('Opponent Country')[['Your Score','Opponent Score']].apply(lambda x: (x['Your Score']>x['Opponent Score']).mean()*100).apply(lambda x: round(x,2))
        by_country_against[['Your Score', 'Opponent Score', 'Score Difference','Distance']]=by_country_against[['Your Score', 'Opponent Score', 'Score Difference','Distance']].apply(round)

        return by_country_against
    @staticmethod
    def create_map(df, metric_col):
        lat_col='Your Latitude'
        lon_col='Your Longitude'
        color_=px.colors.sequential.Turbo_r
        if metric_col=='Distance':
            color_=px.colors.sequential.Turbo
            metric_col='Your Distance'
        fig = px.scatter_geo(
            df,
            lat=lat_col,
            lon=lon_col,
            color=metric_col,
            color_continuous_scale=color_,
            projection="mercator", 
        )
        if 'marker_size' not in st.session_state:
            st.session_state['marker_size']=4
        fig.update_traces(marker=dict(size=st.session_state['marker_size'])) 
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig)
        st.slider('Marker Size',min_value=1,max_value=10,value=st.session_state['marker_size'] ,step=1,key='marker_size')
    @staticmethod
    def create_line_chart(df,  metric_col,date_option):
        date_col='Date'
        
        df['Date'] = pd.to_datetime(df['Date'])
        if date_option == 'Week':
            group_format = 'W'
        elif date_option == 'Month':
            group_format = 'M'
        else:  # Yearly
            group_format = 'Y'

        df['Group'] = df[date_col].dt.to_period(group_format).apply(lambda r: r.start_time)
        df_grouped=df.groupby(by='Group')[metric_col].mean()
        
        fig = px.line(df_grouped,  y=metric_col,markers=True )
        fig.update_layout(bargap=0.1, xaxis_title=date_option, yaxis_title=metric_col)
        fig.update_layout(title_text=f"{metric_col}")
        st.plotly_chart(fig, use_container_width=True)
    @staticmethod
    def scatter_scores(df,col_a,col_b,show_avg_lines,color=None):
        # send with index as countries
        labels_ = df.index.map(lambda x: x if df.index.get_loc(x) % 5 == 0 else "")
        fig=px.scatter(data_frame=df.reset_index(),x=col_a,y=col_b,text=labels_, hover_name= 'Country',color=color,color_continuous_scale='RdBu')
        fig.update_layout(coloraxis_colorbar_title_side='bottom')
        fig.update_coloraxes(colorbar_title_text='',colorbar_xpad=0,colorbar_thickness=5)
        # fig.update_coloraxes(reversescale=True)
        if(show_avg_lines):
            fig.add_shape(
                        type="line",
                        x0=df[col_a].min(), y0=df[col_b].mean(), x1=df[col_a].max(), y1=df[col_b].mean(),
                        line=dict(color='green',width=2, dash="dot"),
                        xref="x", yref="y"
                    )
            # vertical line
            fig.add_shape(
                        type="line",
                        x0=df[col_a].mean(), y0=df[col_b].min(), x1=df[col_a].mean(), y1=df[col_b].max(),
                        line=dict(color='green',width=2, dash="dot"),
                        xref="x", yref="y"
                    )
        fig.update_traces(textposition='top center')
        # fig.update_layout(yaxis_range=[0,5000])
        # fig.update_layout(xaxis_range=[0,5000])
        size_=600
        fig.update_layout(width=size_, height=size_)
        st.plotly_chart(fig,use_container_width=False)
    @staticmethod
    def create_line_chart_games_played(df,  date_option):
        date_option=st.radio('A',('Week','Month','Year'),horizontal=True,label_visibility='hidden')
        date_col='Date'
        metric_col='Games Played'
        df[metric_col]=df['Game Id']
        df['Date'] = pd.to_datetime(df['Date'])
        if date_option == 'Week':
            group_format = 'W'
        elif date_option == 'Month':
            group_format = 'M'
        else:  # Yearly
            group_format = 'Y'
        df.loc[:,'Group'] = df[date_col].dt.to_period(group_format).apply(lambda r: r.start_time)
        df_grouped=df.groupby(by='Group')[metric_col].nunique()
        
        fig = px.line(df_grouped,  y=metric_col,markers=True )
        fig.update_layout( xaxis_title=date_option, yaxis_title=metric_col)
        # fig.update_layout(title_text=f"Games Played")
        st.plotly_chart(fig, use_container_width=True)
    @staticmethod
    def scatter_by_game_type(top_n_countries,df,col_a,col_b,metric_col,show_avg_lines,color=None):
        
        df=df[df['Country'].isin(top_n_countries.index)]
        
        df_a=  df if col_a=='Moving' else  df[~df['Moving']] if col_a=='No Move' else df[(~df['Moving'])&(~df['Zooming'])]
        df_b=  df if col_b=='Moving' else  df[~df['Moving']] if col_b=='No Move' else df[(~df['Moving'])&(~df['Zooming'])]
        
        
        if metric_col=='Number of Rounds':
            metric_col='Round Number'
            df_a=df_a.groupby('Country')[metric_col].count()
            df_b=df_b.groupby('Country')[metric_col].count()
        else:
            if metric_col=='Distance':
                metric_col='Your Distance'
            df_a=df_a.groupby('Country')[metric_col].mean()
            df_b=df_b.groupby('Country')[metric_col].mean()
        
        if col_a==col_b:
            col_a=col_a+' A'
            col_b=col_b+' B'
            
        df_a.rename(col_a,inplace=True)
        df_b.rename(col_b,inplace=True)
        df=pd.concat([df_a,df_b],axis=1)
        
        labels_ = df.index.map(lambda x: x if df.index.get_loc(x) % 5 == 0 else "")
        fig=px.scatter(data_frame=df.reset_index(),x=col_a,y=col_b,text=labels_, hover_name= 'Country',color=color,color_continuous_scale='RdBu')
        fig.update_layout(coloraxis_colorbar_title_side='bottom')
        fig.update_coloraxes(colorbar_title_text='',colorbar_xpad=0,colorbar_thickness=5)
        if(show_avg_lines):
            fig.add_shape(
                        type="line",
                        x0=df[col_a].min(), y0=df[col_b].mean(), x1=df[col_a].max(), y1=df[col_b].mean(),
                        line=dict(color='green',width=2, dash="dot"),
                        xref="x", yref="y"
                    )
            # vertical line
            fig.add_shape(
                        type="line",
                        x0=df[col_a].mean(), y0=df[col_b].min(), x1=df[col_a].mean(), y1=df[col_b].max(),
                        line=dict(color='green',width=2, dash="dot"),
                        xref="x", yref="y"
                    )
        fig.update_traces(textposition='top center')
        # fig.update_layout(yaxis_range=[0,5000])
        # fig.update_layout(xaxis_range=[0,5000])
        size_=600
        fig.update_layout(width=size_, height=size_)
        st.plotly_chart(fig,use_container_width=False)
    

st.title('Welcome to Duels Analyzer')
st.text('I created this tool to analyse my rated duel games on Geoguessr. I hope you find it helpful.')
st.text('It needs your _ncfa token to get your games history and data. Your token is not sent anywhere neither it is saved anywhere. You can check the source code, it is open source.')
with st.expander("For any questions/suggestions, message me here"):
    st.page_link("http://a-azeem.bsky.social",label='Bluesky')
    st.page_link('http://reddit.com/u/brartheonnerd', label='Reddit')
    st.page_link("http://twitter.com/azeemstweet",label="Twitter")
with st.form('input_token'):
    _ncfa = st.text_input("Enter _ncfa token:")
    col1,col2, col3=st.columns(3)
    with col1:
        st.link_button('How to get your _ncfa token',"https://github.com/SafwanSipai/geo-insight?tab=readme-ov-file#getting-your-_ncfa-cookie")
    with col3:
        submitted_token = st.form_submit_button("Enter")
    
if 'submitted_token' not in st.session_state:
    st.session_state['submitted_token']=False

if  (submitted_token or st.session_state['submitted_token']) and _ncfa:
    st.session_state['submitted_token']=True
    session=helpers.get_session(_ncfa)
    player_data=helpers.get_player_data(session)
    if player_data!={}:
        my_player_Id=player_data['id']
        st.write(f'Hello {player_data['nick']} (id {player_data['id']}), extracting your game tokens...')
    if 'duel_tokens' not in st.session_state:
        st.session_state['duel_tokens']=[]
        with st.spinner("", show_time=True):
            duel_tokens=helpers.get_duel_tokens(session)
        st.session_state['duel_tokens']=duel_tokens
    else:
        duel_tokens=st.session_state['duel_tokens']
    st.write(f'Found {len(duel_tokens)} rated duels.')
    
    
    st.write(f'To retrive all {len(duel_tokens)} games, it will take around {60*len(duel_tokens)/500} seconds.')
    st.markdown('I recommend you choose **All**, it will take some time but after that, you can analyse your games withouth any loading.')
    retrieval_option = st.radio(
            "Retrieval Option:",
            # ("Retrieve All", "Retrieve Recent", "Retrieve by Date"),
            ("Retrieve All", "Retrieve Recent"),
            horizontal=False,
            label_visibility="collapsed",
        )
    with st.form("retrieval_form",border=False):
        if retrieval_option == "Retrieve Recent":
            recent_count = st.slider("Recent Games:", 1, len(duel_tokens), round(len(duel_tokens)/2) ) 
        # elif retrieval_option == "Retrieve by Date":
        #     today = datetime.date.today()
        #     start_date = today - datetime.timedelta(days=7)
        #     date_range = st.date_input("Select a date range", (start_date, today),format="DD/MM/YYYY")
        else:
            recent_count = None
            date_range = None
        submitted_1 = st.form_submit_button("Retrieve")
    if 'submitted_1' not in st.session_state:
        st.session_state['submitted_1']=False
    if st.session_state['submitted_1'] or submitted_1:
        st.session_state['submitted_1']=True
        if retrieval_option == "Retrieve All":
            st.write("Retrieving all games' data...")
        elif retrieval_option == "Retrieve Recent":
            st.write(f"Retrieving {recent_count} recent games...")
            duel_tokens=duel_tokens[:recent_count]
        # else:
            # st.write(f"Retrieving games between {date_range[0]} and {date_range[1]}...")
            # to do the whole retrival  by date thing
        data_dict={}
        if len(duel_tokens)>0:
            if 'data_dict' not in st.session_state:
                st.session_state['data_dict']={}
                loading_bar=st.progress(0)
                data_dict=helpers.get_duels(session,duel_tokens,my_player_Id,loading_bar)
                st.success('Done')
                st.session_state['data_dict']=data_dict
            else:
                data_dict=st.session_state['data_dict']
        df=pd.DataFrame()
        df=pd.DataFrame(data_dict)
        if not df.empty:
            df=helpers.datetime_processing(df)

        submitted=False
        option = st.radio(
            'How many games you want to analyze?',
            ('All', 'Recent games','By Date'))
        if option == 'Recent games':
            with st.form("option_form"):
                slider_value = st.slider("Select how many recent games you want to analyse:",min_value= 1,max_value= len(duel_tokens),value=len(duel_tokens))
                submitted = st.form_submit_button("Submit")
        
        elif option == 'By Date':
            with st.form("option_form"):
                today = datetime.date.today()
                start_date = today - datetime.timedelta(days=7)
                date_range = st.date_input("Select a date range", (start_date, today),format="DD/MM/YYYY")
                submitted = st.form_submit_button("Submit")
        else:
            with st.form("option_form"):
                submitted=st.form_submit_button("Submit")
        if 'submitted' not in st.session_state:
            st.session_state['submitted']=False
        if (st.session_state['submitted'] or submitted) and not df.empty:
            st.session_state['submitted']=True
            df_filtered=pd.DataFrame()
            if option == 'By Date':
                df_filtered=df[(df['Date']>=date_range[0]) & (df['Date']<=date_range[1])].copy()
                st.write(f"Found {df_filtered['Game Id'].nunique()} games")
            elif option == 'Recent games':
                df['Running Total'] = (df['Game Id'] != df['Game Id'].shift()).cumsum()
                df_filtered=df[df['Running Total']<=slider_value].copy()
            else:
                df_filtered=df.copy()
            # st.write(df_filtered)
            by_country=helpers.groupby_country(df_filtered)
            top_n = st.slider('Select how many countries you want to see (by round count):',min_value=1,max_value=len(by_country),value=20,step=1,help='This helps filter out the countries that occur very rarely.')
            top_n_countries=by_country.sort_values(by='Number of Rounds',ascending=False).head(top_n)
            
            if not df_filtered.empty:
                st.markdown('### Summary')
                with st.expander(""):    
                    col1, col2=st.columns(2)
                    with col1:
                        st.markdown(f'# {round(df_filtered['Win Percentage'].mean(),2)} %')
                        st.write(f"Win Rate")
                    with col2:
                        st.markdown(f'# {df_filtered.iloc[0]["Your Rating"]}')
                        st.write(f"Duels Rating")
                        
                    best_country_by_win_rate=top_n_countries.sort_values(by='Win Percentage',ascending=False).reset_index().head(1)
                    worst_country_by_win_rate=top_n_countries.sort_values(by='Win Percentage',ascending=True).reset_index().head(1)
                    col1, col2=st.columns(2)
                    with col1:
                        st.markdown(f'# {best_country_by_win_rate.iloc[0]['Country']}')
                        st.write(f"Best win rate:\t{best_country_by_win_rate.iloc[0]['Win Percentage']} %")
                    with col2:
                        st.markdown(f'# {worst_country_by_win_rate.iloc[0]['Country']}')
                        st.write(f"Worst win rate:\t{worst_country_by_win_rate.iloc[0]['Win Percentage']} %")
                    col1, col2=st.columns(2)
                    with col1:
                        st.markdown(f'# {round(df_filtered['Your Score'].mean())}')
                        st.write(f"Average Score")
                    with col2:
                        st.markdown(f'# {round(df_filtered['Your Distance'].mean())}')
                        st.write(f"Average Distance (km)")
                        
                    date_option = st.radio("A",("Week", "Month", "Year"),horizontal=True,label_visibility="collapsed",key='98465')
                    col1, col2=st.columns(2)
                    with col1:
                        helpers.create_line_chart(df_filtered,'Your Rating',date_option)
                    with col2:
                        helpers.create_line_chart(df_filtered,'Your Score',date_option)
                    col1, col2=st.columns(2)
                    with col1:
                        helpers.create_line_chart(df_filtered,'Score Difference',date_option)
                    with col2:
                        helpers.create_line_chart(df_filtered,'Win Percentage',date_option)
                
                st.markdown('### Detailed Analysis')
                with st.expander(""):
                    metric=st.radio('Choose a metric:',('Score','Distance','Score Difference','Win Percentage'))
                    if metric=='Score':
                        metric_col='Your Score'
                    else:
                        metric_col=metric
                    
                    st.markdown(f'#### Average {metric} by Country')
                    helpers.display_country_scores_map(top_n_countries.reset_index(), "Country", metric_col)
                    
                    helpers.sorted_bar_chart(top_n_countries,'Country',metric_col)

                    st.markdown(f'#### Average {metric} by Rounds')
                    by_round=helpers.groupby_round(df_filtered)
                    helpers.sorted_bar_chart(by_round,'Round Number',metric_col)
                    
                    st.markdown(f'#### Average {metric} by Time Periods')
                    helpers.create_binned_histogram(df_filtered,metric_col)
                    
                    st.markdown(f'#### Average {metric} against players from other Countries')
                    by_country_against=helpers.groupby_country_against(df_filtered)
                    helpers.sorted_bar_chart(by_country_against,'Opponent Country',metric_col)
                    
                    st.markdown(f'#### All your guesses, colored by {metric}')
                    helpers.create_map(df_filtered,metric_col)
                    
                    st.markdown(f'#### {metric} distribution by Country')
                    df_filtered_only_top_countries=df_filtered[df_filtered.reset_index()['Country'].isin(top_n_countries.reset_index()['Country'].tolist())]
                    metric_for_box= metric_col if metric_col !='Distance' else 'Your Distance'
                    fig=px.box(data_frame=df_filtered_only_top_countries.reset_index(),x=metric_for_box,y='Country')
                    st.plotly_chart(fig,help='If this is feels cramped, try decreasing the top country parameter at the top')

                st.markdown(f'### Comparisons')
                with st.expander(""):
                    st.markdown('#### Comparison between different metrices')
                    col1, col2,col3 = st.columns(3)
                    options = ['Your Score', 'Opponent Score', 'Score Difference', 'Win Percentage', 'Distance','Number of Rounds']
                    with col1:
                        choice1 = st.selectbox("Metric 1", options, index=0) 
                    with col2:
                        choice2 = st.selectbox("Metric 2", options, index=3) 
                    with col3:
                        choice3 = st.selectbox("Color by", options, index=5) 
                    show_avg_lines=st.checkbox('Show average lines',value=True)
                    helpers.scatter_scores(top_n_countries,choice1,choice2,show_avg_lines,color=choice3)            
                    
                    st.markdown('#### Comparison between different duel types')
                    col1, col2,col3 = st.columns(3)
                    options_game_type = ['Moving', 'No Move', 'NMPZ']
                    with col1:
                        choice1 = st.selectbox("Duel Type 1", options_game_type, index=0) 
                    with col2:
                        choice2 = st.selectbox("Duel Type 2", options_game_type, index=1) 
                    with col3:
                        choice3 = st.selectbox("Metric", options, index=0) 
                    show_avg_lines=st.checkbox('Show average lines',value=True,key='sdfssc')
                    helpers.scatter_by_game_type(top_n_countries,df_filtered,choice1,choice2,choice3,show_avg_lines)            
                    
                st.markdown('### Other Analysis')
                with st.expander(''):
                    st.markdown('#### Round Count by Country')
                    helpers.sorted_bar_chart(by_country,'Country','Number of Rounds')
                    st.markdown('#### Round Count')
                    helpers.sorted_bar_chart(by_round,'Round Number','Number of Rounds')
                    st.markdown('#### Number of games played')
                    helpers.create_line_chart_games_played(df_filtered,'Week')
                    
                    st.markdown('#### Complete extracted data (Download for your own analysis)')
                    st.write(df_filtered)
                


