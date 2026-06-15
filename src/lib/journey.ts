import { journeyStages } from '../data/journeyStages';
import { highlightCode } from './highlight';

export async function buildHighlightedJourneyStages() {
  return Promise.all(
    journeyStages.map(async ({ title, detail, code }) => ({
      title,
      detail,
      codeHtml: await highlightCode(code),
    })),
  );
}
