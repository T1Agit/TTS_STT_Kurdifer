# Kurdish STT Post-Processing

This document describes the Kurdish STT post-processing feature that automatically corrects common transcription errors made by the facebook/mms-1b-all model.

## Overview

The post-processing system consists of three components:

1. **kurdish_dictionary.py** - A comprehensive dictionary of 779+ Kurdish word corrections
2. **kurdish_postprocessor.py** - The post-processing engine that applies corrections
3. **Updated kurdish_stt_service.py** - Integration into the STT service

## Features

### Automatic Correction of Common Mistakes

The system corrects typical STT errors including:

- **Missing special characters**: ê, î, û, ç, ş
- **Vowel confusion**: e/ê, i/î, u/û
- **Consonant confusion**: c/ç, s/ş
- **Case preservation**: Maintains original casing pattern

### Comprehensive Dictionary Coverage

The dictionary includes 779+ corrections across all major word categories:

| Category | Examples |
|----------|----------|
| Greetings | silav, merheba, rojbaş, şevbaş, spas |
| Pronouns | ez, tu, ew, em, hûn, min, te, wî, wê |
| Questions | çi, kî, çawa, kengê, çend, çima |
| Verbs | hatin, çûn, bûn, kirin, xwestin, zanîn |
| Family | dê, bav, bira, xwişk, mam, xal, kur, keç |
| Numbers | yek, du, sê, çar, pênc, şeş, heft, heşt |
| Time | roj, şev, sibê, êvar, nîvro, îro, duh |
| Nature | av, erd, ezman, stêr, çiya, deryâ, gol |
| Body | ser, çav, guh, dev, dest, pê, dil |
| Food | nan, goşt, şîr, penêr, çay, sebze |
| Adjectives | baş, xirab, mezin, biçûk, xweş, germ |
| Prepositions | li, di, bi, ji, bo, ber, pêş, paş |
| Education | dibistan, mamoste, xwendekar, pirtûk |
| Colors | sor, reş, spî, zer, kesk, şîn |
| Animals | se, pisîng, hesp, bizin, mih |

## Usage

### In Code

The post-processor is automatically integrated into `KurdishSTTService`:

```python
from kurdish_stt_service import KurdishSTTService

# Initialize service (post-processor loads automatically)
stt_service = KurdishSTTService()

# Transcribe audio
result = stt_service.transcribe_from_file('audio.mp3')

# Access both raw and corrected transcriptions
print(f"Raw: {result['raw_text']}")
print(f"Corrected: {result['text']}")
```

### Response Format

The STT service now returns both raw and corrected text:

```python
{
    'raw_text': 'silav cawa yi',      # Original STT output
    'text': 'silav çawa yi',           # Post-processed output
    'language': 'kmr-script_latin',
    'sample_rate': 16000,
    'duration': 2.5
}
```

### Standalone Usage

You can also use the post-processor independently:

```python
from kurdish_postprocessor import KurdishPostProcessor

processor = KurdishPostProcessor()

# Correct text
raw = "tu cend sali yi"
corrected = processor.correct_transcription(raw)
print(corrected)  # Output: "tu çend sali yi"

# Get correction statistics
stats = processor.get_correction_stats(raw, corrected)
print(f"Corrected {stats['words_corrected']} out of {stats['total_words']} words")
```

## Examples

### Greetings
```
Input:  "silav cawa yi"
Output: "silav çawa yi"
```

### Responses
```
Input:  "ez bas im"
Output: "ez baş im"
```

### Questions
```
Input:  "nave te ci ye"
Output: "navê te çi ye"

Input:  "cend sali yi"
Output: "çend salî yî"
```

### Numbers
```
Input:  "car se penc ses"
Output: "çar sê pênc şeş"
```

### Family Words
```
Input:  "de bav bira xwisk"
Output: "dê bav bira xwişk"
```

### Food
```
Input:  "nan gost sir cay"
Output: "nan goşt şîr çay"
```

## Testing

Run the comprehensive test suite:

```bash
python test_kurdish_postprocessing.py
```

The test suite includes:
- Dictionary size verification (779+ entries)
- Special character corrections
- Basic functionality tests
- Case preservation tests
- Word category coverage
- Integration simulation

Run the demo to see examples:

```bash
python demo_kurdish_postprocessing.py
```

## Implementation Details

### Dictionary Structure

The dictionary is a simple Python dict mapping incorrect → correct forms:

```python
KURDISH_CORRECTIONS = {
    "cawa": "çawa",      # c → ç
    "xer": "xêr",        # e → ê
    "sir": "şîr",        # s → ş, i → î
    "bun": "bûn",        # u → û
    # ... 779+ total corrections
}
```

### Case Preservation

The post-processor intelligently preserves the original case pattern:

```python
"Cawa" → "Çawa"        # Capitalized
"CAWA" → "ÇAWA"        # All uppercase
"cawa" → "çawa"        # Lowercase
```

### Word-Level Processing

The system processes text word-by-word while preserving:
- Word boundaries
- Punctuation
- Spacing
- Case patterns

## Performance

- **Dictionary size**: 779 corrections
- **Memory footprint**: ~50KB (dictionary)
- **Processing speed**: Near-instantaneous for typical sentences
- **Case handling**: Automatic with pattern preservation

## Future Enhancements

Potential improvements for future versions:

1. **Context-aware corrections**: Use surrounding words to improve accuracy
2. **Confidence-based application**: Only apply corrections when STT confidence is low
3. **Learning from user feedback**: Expand dictionary based on actual usage
4. **Multi-word phrase corrections**: Handle longer common expressions
5. **Regional dialect support**: Add support for different Kurdish dialects

## Contributing

To add more words to the dictionary:

1. Edit `kurdish_dictionary.py`
2. Add entries to `KURDISH_CORRECTIONS` dict
3. Follow the format: `"wrong": "correct"`
4. Include both lowercase and capitalized versions if needed
5. Run tests to verify: `python test_kurdish_postprocessing.py`

## License

Same as the main project.
