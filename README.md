# semantic-search

nlp model , use to semantic-search write with python at Arbic data

Data columns: 
| IllnessID | ill | preprocessed_text |

IllnessID: row identifier    
ill: illness name (come from business "My Frind" )
preprocessed_text: illness description after cleaning preprocessed

I thinking at the semantic search as multi-class classification problem
Traditional machine learning algorithm was used (Train Support Vector Machine)
I use IF-IDf to convert the text form to numerical form ("Embedding") 

Cleaning

	-remove all non-Arbic letters (punctuation , numbers , etc ... )
	-make spell checking using fuzzy search algorithm that try to find the very similar match from my data 
 
Gpt help me to write comment and some of the code. but i learned alot so i will make refactor  to the model 


