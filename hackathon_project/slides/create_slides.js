const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

// Create presentation
let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'Amin Nabavi';
pres.title = 'Lineup Cohesion: A Network-Based Approach to Starting Eleven Optimization';

// Color palette - Soccer Analytics theme
const COLORS = {
    primary: "1E3A5F",      // Deep navy
    secondary: "FFFFFF",     // White
    accent: "2E7D32",        // Soccer green
    highlight: "F4A825",     // Golden amber
    text: "1E293B",          // Dark slate
    textLight: "64748B",     // Muted gray
    bgLight: "F8FAFC",       // Off-white
    bgDark: "0F172A",        // Near black
};

// ============================================================
// SLIDE 1: Title
// ============================================================
let slide1 = pres.addSlide();
slide1.background = { color: COLORS.bgDark };

// Title
slide1.addText("Lineup Cohesion", {
    x: 0.5, y: 1.8, w: 9, h: 1,
    fontSize: 48, fontFace: "Calibri", bold: true,
    color: COLORS.secondary, margin: 0
});

slide1.addText("A Network-Based Approach to Starting Eleven Optimization", {
    x: 0.5, y: 2.7, w: 9, h: 0.6,
    fontSize: 24, fontFace: "Calibri",
    color: COLORS.highlight, margin: 0
});

// Author & event
slide1.addText("Amin Nabavi", {
    x: 0.5, y: 4.0, w: 4, h: 0.4,
    fontSize: 18, fontFace: "Calibri", bold: true,
    color: COLORS.secondary, margin: 0
});

slide1.addText("NEU Soccer Data Analytics Hackathon | February 2026", {
    x: 0.5, y: 4.4, w: 6, h: 0.4,
    fontSize: 14, fontFace: "Calibri",
    color: COLORS.textLight, margin: 0
});

// Accent line
slide1.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.5, w: 2, h: 0.06,
    fill: { color: COLORS.accent }
});

// Data source badge
slide1.addText("Bundesliga 2023/24 · IMPECT Open Data · 306 Matches", {
    x: 0.5, y: 5.1, w: 6, h: 0.3,
    fontSize: 11, fontFace: "Calibri",
    color: COLORS.textLight, margin: 0
});

// ============================================================
// SLIDE 2: Research Question
// ============================================================
let slide2 = pres.addSlide();
slide2.background = { color: COLORS.bgLight };

slide2.addText("Research Question", {
    x: 0.5, y: 0.4, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Calibri", bold: true,
    color: COLORS.primary, margin: 0
});

// Main question box
slide2.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.2, w: 9, h: 1.4,
    fill: { color: COLORS.primary }
});

slide2.addText("Can we quantify team cohesion from passing networks\nto predict match outcomes and optimize lineups?", {
    x: 0.7, y: 1.35, w: 8.6, h: 1.2,
    fontSize: 22, fontFace: "Calibri", bold: true,
    color: COLORS.secondary, align: "center", valign: "middle", margin: 0
});

// Why it matters - 3 cards
const reasons = [
    { title: "Stakeholder Value", desc: "Coaches need objective metrics beyond intuition for lineup decisions" },
    { title: "Network Science", desc: "Passing patterns encode team chemistry that box scores miss" },
    { title: "Actionable Insight", desc: "Identify key hubs, optimal pairings, and substitution strategies" }
];

reasons.forEach((r, i) => {
    const xPos = 0.5 + i * 3.1;
    
    // Card background
    slide2.addShape(pres.shapes.RECTANGLE, {
        x: xPos, y: 2.9, w: 2.9, h: 2.3,
        fill: { color: COLORS.secondary },
        shadow: { type: "outer", blur: 3, offset: 2, angle: 45, opacity: 0.15 }
    });
    
    // Accent bar
    slide2.addShape(pres.shapes.RECTANGLE, {
        x: xPos, y: 2.9, w: 2.9, h: 0.08,
        fill: { color: COLORS.accent }
    });
    
    slide2.addText(r.title, {
        x: xPos + 0.15, y: 3.1, w: 2.6, h: 0.5,
        fontSize: 14, fontFace: "Calibri", bold: true,
        color: COLORS.primary, margin: 0
    });
    
    slide2.addText(r.desc, {
        x: xPos + 0.15, y: 3.6, w: 2.6, h: 1.4,
        fontSize: 12, fontFace: "Calibri",
        color: COLORS.text, margin: 0
    });
});

// ============================================================
// SLIDE 3: Methodology - Cohesion Metric
// ============================================================
let slide3 = pres.addSlide();
slide3.background = { color: COLORS.bgLight };

slide3.addText("Cohesion Metric: Four Components", {
    x: 0.5, y: 0.4, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Calibri", bold: true,
    color: COLORS.primary, margin: 0
});

// Formula box
slide3.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.1, w: 9, h: 0.7,
    fill: { color: COLORS.bgDark }
});

slide3.addText("Cohesion = 0.50·Connectivity + 0.25·Chemistry + 0.15·Hub + 0.10·Progression", {
    x: 0.5, y: 1.15, w: 9, h: 0.6,
    fontSize: 16, fontFace: "Consolas",
    color: COLORS.highlight, align: "center", valign: "middle", margin: 0
});

// Four component cards - 2x2 grid
const components = [
    { name: "Connectivity", weight: "50%", desc: "Network density + clustering coefficient", corr: "r = +0.785***", color: "2E7D32" },
    { name: "Chemistry", weight: "25%", desc: "Strength of midfield→attack connections", corr: "r = +0.448*", color: "1976D2" },
    { name: "Hub Dependence", weight: "15%", desc: "Gini coefficient (star-player reliance)", corr: "r = +0.714***", color: "7B1FA2" },
    { name: "Progression", weight: "10%", desc: "Pre-shot pass ratio in network edges", corr: "r = +0.133", color: "E65100" }
];

components.forEach((c, i) => {
    const xPos = 0.5 + (i % 2) * 4.6;
    const yPos = 2.0 + Math.floor(i / 2) * 1.7;
    
    // Card
    slide3.addShape(pres.shapes.RECTANGLE, {
        x: xPos, y: yPos, w: 4.4, h: 1.5,
        fill: { color: COLORS.secondary },
        shadow: { type: "outer", blur: 2, offset: 1, angle: 45, opacity: 0.1 }
    });
    
    // Left accent
    slide3.addShape(pres.shapes.RECTANGLE, {
        x: xPos, y: yPos, w: 0.08, h: 1.5,
        fill: { color: c.color }
    });
    
    // Component name + weight
    slide3.addText(c.name, {
        x: xPos + 0.2, y: yPos + 0.1, w: 2.5, h: 0.4,
        fontSize: 16, fontFace: "Calibri", bold: true,
        color: COLORS.primary, margin: 0
    });
    
    slide3.addText(c.weight, {
        x: xPos + 3.5, y: yPos + 0.1, w: 0.7, h: 0.4,
        fontSize: 14, fontFace: "Calibri", bold: true,
        color: c.color, align: "right", margin: 0
    });
    
    // Description
    slide3.addText(c.desc, {
        x: xPos + 0.2, y: yPos + 0.5, w: 4, h: 0.5,
        fontSize: 11, fontFace: "Calibri",
        color: COLORS.text, margin: 0
    });
    
    // Correlation
    slide3.addText(c.corr, {
        x: xPos + 0.2, y: yPos + 1.0, w: 2, h: 0.35,
        fontSize: 11, fontFace: "Consolas",
        color: COLORS.textLight, margin: 0
    });
});

// Note
slide3.addText("Weights empirically optimized from correlation with season points (n=18 teams)", {
    x: 0.5, y: 5.2, w: 9, h: 0.3,
    fontSize: 10, fontFace: "Calibri", italic: true,
    color: COLORS.textLight, margin: 0
});

// ============================================================
// SLIDE 4: Key Insight - Hub Dependence Paradox
// ============================================================
let slide4 = pres.addSlide();
slide4.background = { color: COLORS.bgLight };

slide4.addText("Key Insight: The Hub Dependence Paradox", {
    x: 0.5, y: 0.4, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Calibri", bold: true,
    color: COLORS.primary, margin: 0
});

// Left column - Finding
slide4.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.2, w: 4.4, h: 3.8,
    fill: { color: COLORS.secondary },
    shadow: { type: "outer", blur: 3, offset: 2, angle: 45, opacity: 0.1 }
});

slide4.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.2, w: 4.4, h: 0.08,
    fill: { color: "C62828" }
});

slide4.addText("Initial Expectation", {
    x: 0.7, y: 1.4, w: 4, h: 0.4,
    fontSize: 14, fontFace: "Calibri", bold: true,
    color: "C62828", margin: 0
});

slide4.addText("Balanced teams (equal pass distribution) should perform better.", {
    x: 0.7, y: 1.85, w: 4, h: 0.6,
    fontSize: 12, fontFace: "Calibri",
    color: COLORS.text, margin: 0
});

slide4.addText("Actual Finding", {
    x: 0.7, y: 2.6, w: 4, h: 0.4,
    fontSize: 14, fontFace: "Calibri", bold: true,
    color: COLORS.accent, margin: 0
});

slide4.addText("Star-dependent networks win more. Elite teams funnel play through hub players (Xhaka, Kimmich).\n\nOriginal 'Balance' showed r = -0.714 with points.\n\nAfter inverting → Hub Dependence r = +0.714***", {
    x: 0.7, y: 3.0, w: 4, h: 1.8,
    fontSize: 12, fontFace: "Calibri",
    color: COLORS.text, margin: 0
});

// Right column - Before/After comparison
slide4.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.2, w: 4.4, h: 3.8,
    fill: { color: COLORS.bgDark }
});

slide4.addText("Metric Improvement", {
    x: 5.3, y: 1.4, w: 4, h: 0.4,
    fontSize: 14, fontFace: "Calibri", bold: true,
    color: COLORS.highlight, margin: 0
});

// Before stats
slide4.addText("Before (equal weights)", {
    x: 5.3, y: 1.9, w: 4, h: 0.35,
    fontSize: 11, fontFace: "Calibri",
    color: COLORS.textLight, margin: 0
});

slide4.addText("r = 0.314", {
    x: 5.3, y: 2.2, w: 2, h: 0.5,
    fontSize: 28, fontFace: "Calibri", bold: true,
    color: "EF5350", margin: 0
});

slide4.addText("p = 0.204 (not significant)", {
    x: 5.3, y: 2.7, w: 4, h: 0.3,
    fontSize: 11, fontFace: "Calibri",
    color: COLORS.textLight, margin: 0
});

// After stats
slide4.addText("After (optimized weights)", {
    x: 5.3, y: 3.2, w: 4, h: 0.35,
    fontSize: 11, fontFace: "Calibri",
    color: COLORS.textLight, margin: 0
});

slide4.addText("r = 0.728", {
    x: 5.3, y: 3.5, w: 2, h: 0.5,
    fontSize: 28, fontFace: "Calibri", bold: true,
    color: COLORS.accent, margin: 0
});

slide4.addText("p = 0.0006*** (highly significant)", {
    x: 5.3, y: 4.0, w: 4, h: 0.3,
    fontSize: 11, fontFace: "Calibri",
    color: COLORS.textLight, margin: 0
});

slide4.addText("+132% improvement in predictive power", {
    x: 5.3, y: 4.5, w: 4, h: 0.35,
    fontSize: 12, fontFace: "Calibri", bold: true,
    color: COLORS.highlight, margin: 0
});

// ============================================================
// SLIDE 5: Validation Results
// ============================================================
let slide5 = pres.addSlide();
slide5.background = { color: COLORS.bgLight };

slide5.addText("Validation: Season & Match Level", {
    x: 0.5, y: 0.4, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Calibri", bold: true,
    color: COLORS.primary, margin: 0
});

// Left: Season correlation chart (placeholder - will use image)
slide5.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.1, w: 4.6, h: 3.2,
    fill: { color: COLORS.secondary },
    line: { color: "E2E8F0", width: 1 }
});

slide5.addText("[cohesion_vs_points.png]", {
    x: 0.5, y: 2.3, w: 4.6, h: 0.5,
    fontSize: 12, fontFace: "Calibri", italic: true,
    color: COLORS.textLight, align: "center", margin: 0
});

slide5.addText("Season Level: Cohesion vs Points", {
    x: 0.5, y: 4.4, w: 4.6, h: 0.35,
    fontSize: 12, fontFace: "Calibri", bold: true,
    color: COLORS.text, align: "center", margin: 0
});

// Right: Match level boxplot (placeholder)
slide5.addShape(pres.shapes.RECTANGLE, {
    x: 5.3, y: 1.1, w: 4.2, h: 3.2,
    fill: { color: COLORS.secondary },
    line: { color: "E2E8F0", width: 1 }
});

slide5.addText("[cohesion_by_result.png]", {
    x: 5.3, y: 2.3, w: 4.2, h: 0.5,
    fontSize: 12, fontFace: "Calibri", italic: true,
    color: COLORS.textLight, align: "center", margin: 0
});

slide5.addText("Match Level: Cohesion by Result", {
    x: 5.3, y: 4.4, w: 4.2, h: 0.35,
    fontSize: 12, fontFace: "Calibri", bold: true,
    color: COLORS.text, align: "center", margin: 0
});

// Stats summary
slide5.addText("ANOVA: F = 36.64, p < 0.0001 — Significant difference between Win/Draw/Loss", {
    x: 0.5, y: 5.0, w: 9, h: 0.35,
    fontSize: 12, fontFace: "Calibri",
    color: COLORS.text, align: "center", margin: 0
});

// ============================================================
// SLIDE 6: Leverkusen Case Study
// ============================================================
let slide6 = pres.addSlide();
slide6.background = { color: COLORS.bgLight };

slide6.addText("Case Study: Leverkusen's Undefeated Season", {
    x: 0.5, y: 0.4, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Calibri", bold: true,
    color: COLORS.primary, margin: 0
});

// Stats banner
slide6.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.05, w: 9, h: 0.8,
    fill: { color: COLORS.bgDark }
});

const stats = ["28W", "6D", "0L", "90 pts", "87 GF", "24 GA"];
stats.forEach((s, i) => {
    slide6.addText(s, {
        x: 0.7 + i * 1.5, y: 1.15, w: 1.3, h: 0.6,
        fontSize: 18, fontFace: "Calibri", bold: true,
        color: i === 2 ? COLORS.accent : COLORS.secondary, align: "center", valign: "middle", margin: 0
    });
});

// Key finding: Wirtz vs Xhaka
slide6.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 2.0, w: 4.4, h: 2.8,
    fill: { color: COLORS.secondary },
    shadow: { type: "outer", blur: 2, offset: 1, angle: 45, opacity: 0.1 }
});

slide6.addText("Two Hub Types Discovered", {
    x: 0.7, y: 2.15, w: 4, h: 0.4,
    fontSize: 14, fontFace: "Calibri", bold: true,
    color: COLORS.primary, margin: 0
});

slide6.addText([
    { text: "Granit Xhaka", options: { bold: true, breakLine: true } },
    { text: "Volume Hub — 558 passes with Palacios alone", options: { breakLine: true } },
    { text: "Recycles possession, dictates tempo", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Florian Wirtz", options: { bold: true, breakLine: true } },
    { text: "Attack Hub — Highest max betweenness (0.448)", options: { breakLine: true } },
    { text: "Wirtz→Boniface: 22 pre-shot passes (23.4%)", options: { breakLine: true } }
], {
    x: 0.7, y: 2.6, w: 4, h: 2,
    fontSize: 11, fontFace: "Calibri",
    color: COLORS.text, margin: 0
});

// Network visualization placeholder
slide6.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 2.0, w: 4.4, h: 2.8,
    fill: { color: COLORS.secondary },
    line: { color: "E2E8F0", width: 1 }
});

slide6.addText("[leverkusen_network.png]", {
    x: 5.1, y: 3.1, w: 4.4, h: 0.5,
    fontSize: 12, fontFace: "Calibri", italic: true,
    color: COLORS.textLight, align: "center", margin: 0
});

// Key attacking chain
slide6.addText("The Killer Chain: Xhaka → Wirtz → Boniface → GOAL", {
    x: 0.5, y: 5.0, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Calibri", bold: true,
    color: COLORS.accent, align: "center", margin: 0
});

// ============================================================
// SLIDE 7: Lineup Application
// ============================================================
let slide7 = pres.addSlide();
slide7.background = { color: COLORS.bgLight };

slide7.addText("Application: Lineup Optimization", {
    x: 0.5, y: 0.4, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Calibri", bold: true,
    color: COLORS.primary, margin: 0
});

// Use cases
const useCases = [
    { title: "Starting XI Selection", desc: "Score candidate lineups by predicted cohesion. Select players who maximize connectivity with existing starters." },
    { title: "Substitution Strategy", desc: "Identify which subs preserve hub structure vs. which disrupt it. Avoid removing high-betweenness players." },
    { title: "Transfer Targets", desc: "Project how a new signing integrates: simulate their historical passing patterns into the team's network." },
    { title: "Opponent Analysis", desc: "Identify opponent's hub players to mark/isolate. Disrupting their Xhaka-equivalent drops cohesion." }
];

useCases.forEach((u, i) => {
    const xPos = 0.5 + (i % 2) * 4.6;
    const yPos = 1.2 + Math.floor(i / 2) * 2.0;
    
    slide7.addShape(pres.shapes.RECTANGLE, {
        x: xPos, y: yPos, w: 4.4, h: 1.8,
        fill: { color: COLORS.secondary },
        shadow: { type: "outer", blur: 2, offset: 1, angle: 45, opacity: 0.1 }
    });
    
    slide7.addShape(pres.shapes.RECTANGLE, {
        x: xPos, y: yPos, w: 0.08, h: 1.8,
        fill: { color: COLORS.accent }
    });
    
    slide7.addText(u.title, {
        x: xPos + 0.2, y: yPos + 0.15, w: 4, h: 0.4,
        fontSize: 14, fontFace: "Calibri", bold: true,
        color: COLORS.primary, margin: 0
    });
    
    slide7.addText(u.desc, {
        x: xPos + 0.2, y: yPos + 0.55, w: 4, h: 1.1,
        fontSize: 11, fontFace: "Calibri",
        color: COLORS.text, margin: 0
    });
});

// ============================================================
// SLIDE 8: Conclusions & Limitations
// ============================================================
let slide8 = pres.addSlide();
slide8.background = { color: COLORS.bgDark };

slide8.addText("Conclusions", {
    x: 0.5, y: 0.4, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Calibri", bold: true,
    color: COLORS.secondary, margin: 0
});

// Key takeaways
slide8.addText([
    { text: "1. Network cohesion predicts season performance", options: { bold: true, breakLine: true } },
    { text: "   r = 0.728 (p < 0.001) — explains 53% of variance in points", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "2. Hub dependence > balance for elite teams", options: { bold: true, breakLine: true } },
    { text: "   Star-player reliance is a feature, not a bug", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "3. Different hub types serve different functions", options: { bold: true, breakLine: true } },
    { text: "   Xhaka (volume) vs Wirtz (attack) — both essential", options: { breakLine: true } }
], {
    x: 0.5, y: 1.2, w: 5, h: 2.5,
    fontSize: 14, fontFace: "Calibri",
    color: COLORS.secondary, margin: 0
});

// Limitations box
slide8.addShape(pres.shapes.RECTANGLE, {
    x: 5.5, y: 1.2, w: 4, h: 2.5,
    fill: { color: "1E293B" }
});

slide8.addText("Limitations", {
    x: 5.7, y: 1.35, w: 3.6, h: 0.4,
    fontSize: 14, fontFace: "Calibri", bold: true,
    color: COLORS.highlight, margin: 0
});

slide8.addText([
    { text: "• Single season (n=18 teams)", options: { breakLine: true } },
    { text: "• ~24% passes lack receiver ID", options: { breakLine: true } },
    { text: "• Position data from metadata only", options: { breakLine: true } },
    { text: "• No tracking data (spatial context)", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Future: Multi-season validation,", options: { breakLine: true } },
    { text: "player-level predictions", options: { breakLine: true } }
], {
    x: 5.7, y: 1.8, w: 3.6, h: 2,
    fontSize: 11, fontFace: "Calibri",
    color: COLORS.textLight, margin: 0
});

// Footer
slide8.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.8, w: 10, h: 0.85,
    fill: { color: COLORS.accent }
});

slide8.addText("GitHub: github.com/aminnabavi/soccer-cohesion | Contact: nabavi@carleton.ca", {
    x: 0.5, y: 4.95, w: 9, h: 0.3,
    fontSize: 12, fontFace: "Calibri",
    color: COLORS.secondary, align: "center", margin: 0
});

slide8.addText("AI Disclosure: Claude assisted with code development and slide formatting", {
    x: 0.5, y: 5.25, w: 9, h: 0.25,
    fontSize: 10, fontFace: "Calibri", italic: true,
    color: COLORS.secondary, align: "center", margin: 0
});

// Save presentation
pres.writeFile({ fileName: "Nabavi_Hackathon2026.pptx" })
    .then(() => console.log("Created: Nabavi_Hackathon2026.pptx"))
    .catch(err => console.error(err));
