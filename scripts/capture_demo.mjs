// 用系统 Chrome 驱动轨迹面板，产出 README 用的 demo 截图。
// 前置：先 make serve，然后
//   node scripts/capture_demo.mjs
// 依赖 playwright-core；若项目里没有，可 npm i -D playwright-core 后再跑。
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const BASE = process.env.MOOKI_BASE || "http://127.0.0.1:8000";
const OUT = process.env.OUT_DIR || "docs";
const CHROME =
  process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
mkdirSync(OUT, { recursive: true });

const cases = [
  { file: "demo-refund.png", q: "SO20261018006 我不想要了，能退款吗", h: 1180 },
  { file: "demo-logistics.png", q: "SO20260512001 这个单子到哪了，三天没更新了", h: 940 },
  { file: "demo-complaint.png", q: "我要投诉，再不解决就去12315", h: 760 },
];

const browser = await chromium.launch({ executablePath: CHROME, headless: true });

async function send(page, text) {
  await page.fill("#q", text);
  await page.click("#send");
  // 发送按钮恢复可用即代表本轮 run 结束
  await page.waitForFunction(() => {
    const b = document.querySelector("#send");
    return b && !b.disabled;
  }, { timeout: 20000 });
  await page.waitForTimeout(500);
}

for (const c of cases) {
  const page = await browser.newPage({
    viewport: { width: 1560, height: c.h },
    deviceScaleFactor: 2,
  });
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector("#q");
  await send(page, c.q);
  await page.$eval("#traceBody", (el) => (el.scrollTop = 0));
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}/${c.file}` });
  console.log("captured", c.file);
  await page.close();
}

// 多轮：第二、三句不带单号，展示结构化槽位继承
{
  const page = await browser.newPage({
    viewport: { width: 1560, height: 1000 },
    deviceScaleFactor: 2,
  });
  await page.goto(BASE, { waitUntil: "networkidle" });
  for (const t of [
    "SO20261018006 我不想要了，能退款吗",
    "那钱多久能到账",
    "能退到原来的微信吗",
  ]) {
    await send(page, t);
  }
  await page.screenshot({ path: `${OUT}/demo-multiturn.png` });
  console.log("captured demo-multiturn.png");
  await page.close();
}

await browser.close();
