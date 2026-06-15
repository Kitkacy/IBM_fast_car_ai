# TORCS RL

## How to run

Create and activate the virtual environment, then install the project:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Train with the default non-GUI launch:

```powershell
python main.py --algorithm sac
```

Train with the visible TORCS window:

```powershell
python main.py --algorithm sac --gui
```

Evaluate a saved model:

```powershell
python main.py --evaluate-model runs\corkscrew-sac\models\best_lap_model.zip --algorithm sac --eval-episodes 5
```

Evaluate with the visible TORCS window:

```powershell
python main.py --evaluate-model runs\corkscrew-sac\models\best_lap_model.zip --algorithm sac --eval-episodes 5 --gui
```

Run a SAC sweep:

```powershell
python main.py --algorithm sac --sweep --sweep-count 20
```

## Track selection

The project always launches `../../torcs/torcs/config/raceman/practice.xml`.

The track is chosen inside that XML:

```xml
<section name="Tracks">
  <section name="1">
    <attstr name="name" val="..."/>
    <attstr name="category" val="..."/>
  </section>
</section>
```
