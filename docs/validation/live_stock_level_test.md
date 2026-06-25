# Live Stock Level Test

## Status

The live shelf inspection prototype is working.

## Current capabilities

- Loads trained YOLO segmentation model from `model/`
- Loads enrolled shelf slot reference from `configs/shelf_base_reference.json`
- Runs live camera inference
- Detects visible shelf base area
- Estimates practical stock level
- Detects item type per slot
- Flags mixed item placement
- Displays clean slot-level status on the live camera feed

## Practical stock labels

| Stock Percent | Decision Label |
|---|---|
| 0–20% | Empty / Need Restock ASAP |
| 20–50% | Low Stock / Restock Soon |
| 50–75% | Partial / Light Restocking |
| 75–93% | Almost Full / No Restocking |
| 93–100% | Full / No Restocking |

## Visual color meaning

| Status | Color |
|---|---|
| Empty / Need Restock ASAP | Red |
| Low Stock / Restock Soon | Yellow |
| Partial / Light Restocking | Orange |
| Almost Full / No Restocking | Blue |
| Full / No Restocking | Green |
| Mixed Items | Red |

## Notes

The stock percentage is treated as a practical shelf occupancy signal, not an exact item count.  
The final decision label is what matters for restocking action.
