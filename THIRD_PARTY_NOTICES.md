# Third-Party Notices

The repository's main `LICENSE` is Apache License 2.0. The following notices
identify incorporated third-party source code and the separately licensed base
model used by Spooftral. The MIT notices below remain applicable to their
respective source code, including the local adaptations.

## RawBoost

Source: https://github.com/TakHemlata/RawBoost-antispoofing

Local files: `utils/RawBoost.py` and `utils/data_utils_rawboost.py` contain
code derived from or adapted from the RawBoost implementation. The local
versions have been modified for Spooftral. The original project is licensed
under the MIT License, with this copyright notice:

Copyright (c) 2021 Hemlata

### RawBoost MIT License

MIT License

Copyright (c) 2021 Hemlata

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ConfidenceIntervals

Source: https://github.com/luferrer/ConfidenceIntervals

Local file: `confidence_intervals/confidence_intervals.py` contains code
derived from or adapted from the original ConfidenceIntervals implementation.
The local version has been modified for Spooftral. The original project is
licensed under the MIT License, with this copyright notice:

Copyright (c) 2023 Luciana Ferrer

### ConfidenceIntervals MIT License

MIT License

Copyright (c) 2023 Luciana Ferrer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Voxtral base model

Spooftral uses Mistral AI's `mistralai/Voxtral-Mini-3B-2507` as its base model:
https://huggingface.co/mistralai/Voxtral-Mini-3B-2507

The model is published under the Apache License 2.0. Its model files are loaded
from the upstream distribution and are not included in this repository.
