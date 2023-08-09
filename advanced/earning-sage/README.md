# 🧙 Earning-Sage
This is an earning call assistant built for investors to help them make better decisions. 

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

