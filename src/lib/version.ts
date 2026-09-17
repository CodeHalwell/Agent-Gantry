/**
 * The release version the documentation site displays.
 *
 * Read from `package.json` so it cannot drift: the site previously hard-coded
 * `0.11.0` in three places and kept advertising it three releases later.
 * `RELEASING.md` bumps `package.json` with the Python version of record.
 */
import pkg from '../../package.json';

export const SITE_VERSION: string = pkg.version;
