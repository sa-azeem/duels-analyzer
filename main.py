import streamlit as st
import random
from math import floor 
import pandas as pd, numpy as np, requests, json
import datetime
import helpers
import plotly.express as px
import altair as alt

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
        if st.session_state['submitted'] or submitted:
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
            


