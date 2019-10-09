# DocAI - Medical documents tagging


## Data collection

Run (it will take few minutes to fetch all the abstracts from PubMed)
```{bash}
python repository/abstracts.py
```

## Modeling and output

Once you fetched abstracts, run the main script

```{bash}
python main.py
```

It will output a file at the root folder named `abstracts_labelled.csv` with labels for each document.





