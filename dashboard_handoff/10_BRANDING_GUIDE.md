# GOLD360 — Branding Guide

## Brand Identity

**GOLD360** is a sovereign economic intelligence platform. The visual identity conveys:
- **Authority** — dark background, gold accents
- **Precision** — clean typography, structured layouts
- **Trust** — consistent color coding, professional styling

## Color Palette

### Primary Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Gold | `#D4AF37` | 212, 175, 55 | Headers, active nav, metric values, brand accent |
| Dark Slate | `#1E293B` | 30, 41, 59 | Card backgrounds, secondary surfaces |
| Dark Background | `#0F172A` | 15, 23, 42 | Page background |

### Secondary Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Light Slate | `#334155` | 51, 65, 85 | Slider backgrounds, subtle surfaces |
| Border | `#2D3748` | 45, 55, 72 | Card borders, dividers |

### Text Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Primary Text | `#F1F5F9` | 241, 245, 249 | Body text, headings |
| Secondary Text | `#94A3B8` | 148, 163, 184 | Subtitles, labels, descriptions |

### Accent Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Green | `#38A169` | 56, 161, 105 | Low risk, positive deltas, success |
| Red | `#E53E3E` | 229, 62, 62 | High risk, negative deltas, errors |
| Amber | `#D69E2E` | 214, 158, 46 | Moderate risk, warnings |
| Orange | `#DD6B20` | 221, 107, 32 | Elevated risk, operational |

### Domain Colors

| Domain | Hex | Usage |
|--------|-----|-------|
| Delivery | `#D4AF37` | Feature group coloring |
| Macro | `#38A169` | Feature group coloring |
| Operational | `#DD6B20` | Feature group coloring |
| Governance | `#805AD5` | Feature group coloring |
| Spatial | `#E53E3E` | Feature group coloring |
| Trade | `#D69E2E` | Feature group coloring |

## Risk Color Mapping

| Risk Level | Score Range | Color | Hex |
|------------|-------------|-------|-----|
| Low | [0.0, 0.25] | Green | `#38A169` |
| Moderate | [0.25, 0.50] | Amber | `#D69E2E` |
| Elevated | [0.50, 0.75] | Orange | `#DD6B20` |
| High | [0.75, 1.0] | Red | `#E53E3E` |

## Typography

| Element | Font | Size | Weight | Color |
|---------|------|------|--------|-------|
| Logo Text | System | 1.8rem | 800 | `#D4AF37` |
| Logo Subtitle | System | 0.9rem | 400 | `#94A3B8` |
| Page Title | System | 1.8rem | 700 | `#D4AF37` |
| Section Header | System | 1.3rem | 600 | `#F1F5F9` |
| Body Text | System | 1rem | 400 | `#F1F5F9` |
| Label | System | 0.85rem | 400 | `#94A3B8` |
| Metric Value | System | 2.2rem | 700 | `#D4AF37` |
| Metric Label | System | 0.8rem | 400 | `#94A3B8` |

## CSS Variables

```css
:root {
    --primary: #D4AF37;
    --secondary: #1E293B;
    --bg-dark: #0F172A;
    --bg-card: #1E293B;
    --bg-light: #334155;
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --accent-green: #38A169;
    --accent-red: #E53E3E;
    --accent-amber: #D69E2E;
    --accent-orange: #DD6B20;
    --border: #2D3748;
}
```

## Card Styling

```css
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 1rem;
}
```

## Responsive Breakpoints

| Breakpoint | Width | Layout |
|------------|-------|--------|
| Mobile | < 640px | Single column, stacked |
| Tablet | 640-1024px | 2-column grid |
| Desktop | > 1024px | Full wide, sidebar |

## Streamlit Theme Config

```toml
[theme]
primaryColor = "#D4AF37"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#F1F5F9"
font = "sans serif"
```
