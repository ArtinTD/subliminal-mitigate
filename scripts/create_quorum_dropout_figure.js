#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected argument: ${arg}`);
    }
    const key = arg.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    args[key] = value;
    i += 1;
  }
  return args;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function escapeXml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function wilsonInterval(hits, n, z = 1.959963984540054) {
  if (n <= 0) {
    throw new Error(`Wilson interval requires n > 0, got ${n}`);
  }
  const p = hits / n;
  const denominator = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denominator;
  const halfWidth =
    (z / denominator) *
    Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [Math.max(0, center - halfWidth), Math.min(1, center + halfWidth)];
}

function truncationHits(summary) {
  if (Number.isInteger(summary.truncation_hits)) {
    return summary.truncation_hits;
  }
  return (
    summary.stop_reasons?.max_new_tokens ??
    summary.stop_reason_counts?.max_new_tokens ??
    0
  );
}

function extractMethod(label, file) {
  const payload = readJson(file);
  const summary = payload.summary;
  if (!summary) {
    throw new Error(`Missing summary in ${file}`);
  }
  const n = summary.n_responses;
  if (n !== 128) {
    throw new Error(`${label} has n=${n}; expected matched n=128`);
  }

  const costKeys = [
    "first_line_eagle",
    "first_line_topaz",
    "first_line_birch",
    "first_line_cobalt",
  ];
  const costs = {};
  for (const key of costKeys) {
    const item = summary.cost_rates?.[key];
    if (!item || item.n_responses !== n) {
      throw new Error(`${label} is missing a matched ${key} count`);
    }
    costs[key.replace("first_line_", "")] = item.hits;
  }

  const stackedCostHits = Object.values(costs).reduce(
    (total, hits) => total + hits,
    0,
  );
  if (stackedCostHits !== summary.any_cost_hits) {
    throw new Error(
      `${label}: stacked costs sum to ${stackedCostHits}, ` +
        `but any-cost count is ${summary.any_cost_hits}`,
    );
  }

  return {
    label,
    source: file,
    n,
    jokeHits: summary.joke_flex_last_hits,
    jokeRate: summary.joke_flex_last_hits / n,
    costHits: costs,
    anyCostHits: summary.any_cost_hits,
    anyCostRate: summary.any_cost_hits / n,
    truncationHits: truncationHits(summary),
  };
}

function rateLabel(rate) {
  return rate.toFixed(2);
}

function errorBar(x, low, high, yScale) {
  const cap = 10;
  const yLow = yScale(low);
  const yHigh = yScale(high);
  return [
    `<line x1="${x}" y1="${yLow}" x2="${x}" y2="${yHigh}" class="error"/>`,
    `<line x1="${x - cap}" y1="${yLow}" x2="${x + cap}" y2="${yLow}" class="error"/>`,
    `<line x1="${x - cap}" y1="${yHigh}" x2="${x + cap}" y2="${yHigh}" class="error"/>`,
  ].join("\n");
}

function legendItem(x, y, fill, label, pattern = null) {
  const fillValue = pattern ? `url(#${pattern})` : fill;
  return [
    `<rect x="${x}" y="${y - 20}" width="42" height="26" rx="1" fill="${fillValue}"/>`,
    `<text x="${x + 54}" y="${y + 2}" class="legend">${escapeXml(label)}</text>`,
  ].join("\n");
}

async function main() {
  const args = parseArgs(process.argv);
  const root =
    args.root ??
    path.join(
      "hyak_results",
      "outputs",
      "composed_joke_explicit_cost",
      "quorum_dropout_m4",
      "pilot128",
    );
  const output =
    args.output ??
    path.join("colm_2026", "figures", "quorum_dropout_results.png");

  const methods = [
    extractMethod("Eagle", path.join(root, "references", "eagle.json")),
    extractMethod("Topaz", path.join(root, "references", "topaz.json")),
    extractMethod("Birch", path.join(root, "references", "birch.json")),
    extractMethod(
      "Cobalt-only",
      path.join(root, "references", "cobalt_cost_only.json"),
    ),
    extractMethod(
      "Merged LoRA",
      path.join(root, "merged_lora", "all_four_equal.json"),
    ),
    extractMethod(
      "Exact min",
      path.join(
        root,
        "generations",
        "m4_three_benefit_one_dropout",
        "min.json",
      ),
    ),
    extractMethod(
      "Quorum q=3",
      path.join(
        root,
        "generations",
        "m4_three_benefit_one_dropout",
        "quorum_q3.json",
      ),
    ),
  ];

  const width = 1580;
  const height = 740;
  const margin = { top: 150, right: 42, bottom: 112, left: 112 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const yMax = 1.05;
  const yScale = (value) =>
    margin.top + plotHeight - (value / yMax) * plotHeight;
  const groupWidth = plotWidth / methods.length;
  const barWidth = 70;
  const barGap = 10;
  const green = "#79C58E";
  const red = "#F08080";
  const redStroke = "#B64C4C";

  const svg = [];
  svg.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="white"/>`,
    `<defs>`,
    `<pattern id="topazPattern" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">`,
    `<rect width="12" height="12" fill="${red}"/>`,
    `<line x1="0" y1="0" x2="0" y2="12" stroke="${redStroke}" stroke-width="4"/>`,
    `</pattern>`,
    `<pattern id="birchPattern" width="13" height="13" patternUnits="userSpaceOnUse">`,
    `<rect width="13" height="13" fill="${red}"/>`,
    `<path d="M0,0 L13,13 M13,0 L0,13" stroke="${redStroke}" stroke-width="2.8"/>`,
    `</pattern>`,
    `<pattern id="cobaltPattern" width="12" height="12" patternUnits="userSpaceOnUse">`,
    `<rect width="12" height="12" fill="${red}"/>`,
    `<circle cx="3" cy="3" r="2.1" fill="${redStroke}"/>`,
    `<circle cx="9" cy="9" r="2.1" fill="${redStroke}"/>`,
    `</pattern>`,
    `<style>`,
    `text { font-family: Arial, Helvetica, sans-serif; fill: #262626; }`,
    `.title { font-size: 40px; font-weight: 500; }`,
    `.axis { font-size: 27px; }`,
    `.ylabel { font-size: 31px; }`,
    `.value { font-size: 24px; }`,
    `.legend { font-size: 24px; }`,
    `.grid { stroke: #D7D7D7; stroke-width: 1.5; stroke-dasharray: 6 5; }`,
    `.axisline { stroke: #B7B7B7; stroke-width: 2; }`,
    `.error { stroke: #1D1D1D; stroke-width: 3; }`,
    `</style>`,
    `</defs>`,
  );

  svg.push(
    `<text x="${width / 2}" y="49" text-anchor="middle" class="title">Quorum recovers shared behavior under incomplete support</text>`,
  );

  const legendY = 101;
  const legendStarts = [238, 515, 705, 930, 1132];
  svg.push(
    legendItem(legendStarts[0], legendY, green, "Final joke"),
    legendItem(legendStarts[1], legendY, red, "Eagle cost"),
    legendItem(legendStarts[2], legendY, red, "Topaz cost", "topazPattern"),
    legendItem(legendStarts[3], legendY, red, "Birch cost", "birchPattern"),
    legendItem(legendStarts[4], legendY, red, "Cobalt cost", "cobaltPattern"),
  );

  for (let tick = 0; tick <= 5; tick += 1) {
    const value = tick / 5;
    const y = yScale(value);
    svg.push(
      `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="grid"/>`,
      `<text x="${margin.left - 18}" y="${y + 9}" text-anchor="end" class="axis">${value.toFixed(1)}</text>`,
    );
  }

  const baseline = yScale(0);
  svg.push(
    `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${baseline}" class="axisline"/>`,
    `<line x1="${margin.left}" y1="${baseline}" x2="${width - margin.right}" y2="${baseline}" class="axisline"/>`,
    `<text transform="translate(39 ${margin.top + plotHeight / 2}) rotate(-90)" text-anchor="middle" class="ylabel">Observed rate</text>`,
  );

  methods.forEach((method, index) => {
    const center = margin.left + groupWidth * (index + 0.5);
    const greenX = center - barGap / 2 - barWidth;
    const redX = center + barGap / 2;

    const jokeY = yScale(method.jokeRate);
    svg.push(
      `<rect x="${greenX}" y="${jokeY}" width="${barWidth}" height="${baseline - jokeY}" fill="${green}"/>`,
    );

    let accumulated = 0;
    const segments = [
      ["eagle", red],
      ["topaz", "url(#topazPattern)"],
      ["birch", "url(#birchPattern)"],
      ["cobalt", "url(#cobaltPattern)"],
    ];
    for (const [costId, fill] of segments) {
      const segmentRate = method.costHits[costId] / method.n;
      if (segmentRate === 0) {
        continue;
      }
      const segmentTop = accumulated + segmentRate;
      const y = yScale(segmentTop);
      const segmentBottom = yScale(accumulated);
      svg.push(
        `<rect x="${redX}" y="${y}" width="${barWidth}" height="${segmentBottom - y}" fill="${fill}"/>`,
      );
      accumulated = segmentTop;
    }

    const [jokeLow, jokeHigh] = wilsonInterval(method.jokeHits, method.n);
    const [costLow, costHigh] = wilsonInterval(method.anyCostHits, method.n);
    svg.push(
      errorBar(greenX + barWidth / 2, jokeLow, jokeHigh, yScale),
      errorBar(redX + barWidth / 2, costLow, costHigh, yScale),
    );

    const jokeLabelY = Math.max(margin.top + 18, yScale(jokeHigh) - 12);
    const costLabelY = Math.max(margin.top + 18, yScale(costHigh) - 12);
    svg.push(
      `<text x="${greenX + barWidth / 2}" y="${jokeLabelY}" text-anchor="middle" class="value">${rateLabel(method.jokeRate)}</text>`,
      `<text x="${redX + barWidth / 2}" y="${costLabelY}" text-anchor="middle" class="value">${rateLabel(method.anyCostRate)}</text>`,
      `<text x="${center}" y="${baseline + 43}" text-anchor="middle" class="axis">${escapeXml(method.label)}</text>`,
    );
  });

  svg.push(`</svg>`);
  const svgText = svg.join("\n");

  fs.mkdirSync(path.dirname(output), { recursive: true });
  const svgOutput = output.replace(/\.png$/i, ".svg");
  const summaryOutput = output.replace(/\.png$/i, ".summary.json");
  fs.writeFileSync(svgOutput, svgText);
  fs.writeFileSync(
    summaryOutput,
    `${JSON.stringify(
      {
        generatedFrom: methods.map((method) => method.source),
        validation: {
          expectedN: 128,
          stackedCostsEqualAnyCost: true,
          intervals: "Wilson 95%",
        },
        methods,
      },
      null,
      2,
    )}\n`,
  );
  await sharp(Buffer.from(svgText)).png().toFile(output);

  console.log(
    JSON.stringify(
      methods.map((method) => ({
        method: method.label,
        n: method.n,
        joke: `${method.jokeHits}/${method.n}`,
        costs: method.costHits,
        anyCost: `${method.anyCostHits}/${method.n}`,
        truncation: `${method.truncationHits}/${method.n}`,
      })),
      null,
      2,
    ),
  );
  console.log(`Wrote ${output}`);
  console.log(`Wrote ${svgOutput}`);
  console.log(`Wrote ${summaryOutput}`);
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exitCode = 1;
});
