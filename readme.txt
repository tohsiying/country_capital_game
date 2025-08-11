# command to run locally
python3 -m streamlit run country_capital_game.py

# open app (that runs locally) in browser
http://localhost:8501/

# register code changes and send these changes to github (after that streamlit cloud automatically redeploys from github)
git add .
git commit -m 'Some text about the change you did'
git push origin main

# open app (that runs remotely in streamlit cloud) in browser
https://country-capital-game.streamlit.app/