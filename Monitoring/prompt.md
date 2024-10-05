# Prompt for Monitoring

I have a streamlit app and I want to add +1 and -1 buttons there. 
Also, I to use postgres for storing the data, so let's add another module for interacting with pg. It should: 
write the question and the answer 
save user feedback if we get anything (probably in a separate table).
Also, I want a docker-compose file for the streamlit app, postgres and Grafana for monitoring.
Also, make data in lancedb persistent, so I don't need to reindex the data every time I run 
add postgres (also with volume mapping)
add a container for the streamlit app (and a docker file for it)