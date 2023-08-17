# 🧙 Earning-Sage
Have you ever thought about joining an earning call and asking questions to these CFOs? That used to be the privilege held by the investors from high-end investment banks such as JP Morgan, Goldman Sachs and Morgan Stanley.

Yet with the capability of LLM and proper techniques around it, not anymore. And if you don’t feel like reading the whole post, feel free to try out a demo [here](https://earningsage.lepton.run/). This demo is created based on the Apple Q2 2023 earning call.

The full documentation can be found [here](https://www.lepton.ai/docs/examples/earning_sage).

## Getting Started

### Step 1 : Setup env
In `main.py` , change line 48 and 49 to the URL with corresponding URL and token. 


### Step 2 : Create a photon
```shell
lep photon create -n earning-sage -m py:main.py
```

### Step 3 : Run the photon
```shell
# Running locally 
lep photon run -n earning-sage --local
# Running remotely, this requies login to lepton.ai 
lep photon push -n earning-sage
lep photon run -n earning-sage
```

