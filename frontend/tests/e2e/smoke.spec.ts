import { test, expect } from '@playwright/test';

const frontendUrl = process.env.FRONTEND_URL ?? 'http://localhost:5173';
const sessionId = process.env.EVAL_SESSION_ID;

test('data analysis agent UI smoke', async ({ page }) => {
  const url = sessionId ? `${frontendUrl}/?session=${encodeURIComponent(sessionId)}` : frontendUrl;
  await page.goto(url, { waitUntil: 'networkidle' });
  await expect(page.getByText(/API online|API offline|Session ready|Safe mode/i).first()).toBeVisible({ timeout: 15000 });
  await page.screenshot({ path: process.env.SMOKE_INITIAL_SCREENSHOT ?? 'test-results/ui-initial.png', fullPage: true });

  const bodyText = await page.locator('body').innerText();
  expect(bodyText.length).toBeGreaterThan(20);

  await page.screenshot({ path: process.env.SMOKE_FINAL_SCREENSHOT ?? 'test-results/ui-smoke.png', fullPage: true });
  const renderableElements =
    (await page.locator('table, svg, canvas, a[download]').count()) +
    (await page.getByRole('button', { name: /download|confirm|apply change|cancel/i }).count()) +
    (await page.getByText(/Confirm|Apply change|Cancel/i).count());
  if (sessionId) {
    expect(renderableElements).toBeGreaterThan(0);
  }
});
