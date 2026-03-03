# Prosperity Sermon Corpus — Style Guide

**Prosperity vs. non-prosperity:** Two distinct color groups so channels are instantly recognizable.

## Group 1 — Prosperity (Purple)

| Color | Hex |
|-------|-----|
| Eggplant | `#3A0CA3` |
| Medium violet | `#7209B7` |
| Soft lilac | `#CDB4DB` |

## Group 2 — Non-Prosperity / Control (Green)

| Color | Hex |
|-------|-----|
| Forest | `#1B4332` |
| Emerald | `#2D6A4F` |
| Mint | `#95D5B2` |

## Neutrals

| Color | Hex | Use |
|-------|-----|-----|
| Chalk white | `#F9F9F9` | Backgrounds |
| Graphite | `#2C2C2C` | Text, borders |

## Channel → Color Mapping

**Prosperity:** joel_osteen, elevation, creflo → Group 1 (eggplant, violet, lilac)  
**Control:** desiring_god, mclean_bible, village_church → Group 2 (forest, emerald, mint)

### CSS Variables

```css
:root {
  --eggplant: #3A0CA3;
  --medium-violet: #7209B7;
  --soft-lilac: #CDB4DB;
  --forest: #1B4332;
  --emerald: #2D6A4F;
  --mint: #95D5B2;
  --chalk-white: #F9F9F9;
  --graphite: #2C2C2C;
}
```
