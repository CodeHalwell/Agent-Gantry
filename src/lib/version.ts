/**
 * The release version the documentation site displays.
 *
 * Read from `package.json` so it cannot drift: the site previously hard-coded
 * `0.11.0` in three places and kept advertising it three releases later.
 * `RELEASING.md` lists `package.json` among the four places a release bumps, and
 * `tests/test_version_consistency.py` fails if it lags `agent_gantry.__version__`.
 */
import pkg from '../../package.json';

export const SITE_VERSION: string = pkg.version;
