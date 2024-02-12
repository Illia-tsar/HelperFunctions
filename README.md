# HelperFunctions | Usage

## Clone the Repository

```bash
git clone https://github.com/Illia-tsar/HelperFunctions.git
cd repository
```

## Create a Python Virtual Environment

```bash
sudo apt-get install python3-venv
```

Now, create a virtual environment:

```bash
python3 -m venv venv
```

## Activate the Virtual Environment

```bash
source venv/bin/activate
```

You should see the virtual environment's name in your terminal prompt, indicating that you are now working within the virtual environment.

## Install Dependencies

Install the project dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Change patch and model 

Navigate to [postprocess.py](https://github.com/Illia-tsar/HelperFunctions/blob/31551174384d6157139a6d4c2458ad4414dd911a/postprocess.py#L164) and set up patch and model paths

## Run the Script

You're all set! Now you can run the script:

```bash
python postprocess.py
```
