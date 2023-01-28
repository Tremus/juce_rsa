const assert = require('assert');

const UINT_MAX = 4294967295;

function countNumberOfBits(n: number) {
    n -= ((n >>> 1) & 0x55555555);
    n = (((n >>> 2) & 0x33333333) + (n & 0x33333333));
    n = (((n >>> 4) + n) & 0x0f0f0f0f);
    n += (n >>> 8);
    n += (n >>> 16);
    return n & 0x3f;
}

function findHighestSetBit(n: number) {
    assert.ok(n != 0);
    n |= (n >>> 1);
    n |= (n >>> 2);
    n |= (n >>> 4);
    n |= (n >>> 8);
    n |= (n >>> 16);
    return countNumberOfBits(n >>> 1);
}

function bitToMask(bit: number) {
    return 1 << (bit & 31);
}

function bitToIndex(bit: number) {
    return bit >>> 5;
}

function sizeNeededToHold(highestBit: number) {
    return (highestBit >>> 5) + 1;
}

function getHexDigitValue(digit: number) {
    let d = digit - '0'.charCodeAt(0);
    if (d < 10) {
        return d;
    }
    d += '0'.charCodeAt(0) - 'a'.charCodeAt(0);

    if (d < 6) {
        return d + 10;
    }
    d += 'a'.charCodeAt(0) - 'A'.charCodeAt(0);

    if (d < 6) {
        return d + 10;
    }
    return -1;
}


function create32bit(num: number) {
    assert.ok(num >= 0);
    assert.ok(num < UINT_MAX);
    const i = new JUCEBigInteger();
    i.highestBit = 31;
    i.negative = num < 0;
    i.preallocated = new Uint32Array([num]);
    i.highestBit = i.getHighestBit();
    return i;
}

class JUCEBigInteger {
    // __slots__ = ('highestBit', 'negative', 'preallocated');
    highestBit: number;
    negative: boolean;
    preallocated: Uint32Array;

    constructor() {
        this.highestBit = -1;
        this.negative = false;
        this.preallocated = new Uint32Array([0]);
    }

    __repr__() {
        return String(this.__int__());
    }
    __int__(): Number {
        return Number(this.toString(10));
    }
    __eq__(other: JUCEBigInteger): boolean {
        return this.compare(other) == 0;
    }
    __ne__(other: JUCEBigInteger): boolean {
        return this.compare(other) != 0;
    }
    __lt__(other: JUCEBigInteger): boolean {
        return this.compare(other) < 0;
    }
    __le__(other: JUCEBigInteger): boolean {
        return this.compare(other) <= 0;
    }
    __gt__(other: JUCEBigInteger): boolean {
        return this.compare(other) > 0;
    }
    __ge__(other: JUCEBigInteger): boolean {
        return this.compare(other) >= 0;
    }
    __neg__(): JUCEBigInteger {
        const result = this.copy();
        result.setNegative(!result.isNegative());
        return result;
    }
    __add__(other: JUCEBigInteger): JUCEBigInteger {
        return this.copy().__iadd__(other);
    }
    __iadd__(other: JUCEBigInteger): this {
        if (other.isNegative()) {
            // this -= -other
            // return this
            return this.__isub__(other.__neg__());
        }

        if (this.isNegative()) {
            if (this.compareAbsolute(other) < 0) {
                const temp = this.copy();
                temp.negate();
                this.copyFrom(other);
                // this -= temp;
                this.__isub__(temp);
            } else {
                this.negate();
                // this -= other;
                this.__isub__(other);
                this.negate();
            }
        } else {
            this.highestBit = Math.max(this.highestBit, other.highestBit) + 1;
            const numInts = sizeNeededToHold(this.highestBit);
            const values = this.ensureSize(numInts);
            const otherValues = other.getValues();
            let remainder = 0;
            for (let i = 0; i < numInts; i++) {
                remainder += values[i];

                if (i < otherValues.length) {
                    remainder += otherValues[i];
                }

                if (remainder > UINT_MAX) {
                    // Take top bytes
                    const hex = remainder.toString(16);
                    values[i] = parseInt(hex.slice(hex.length - 8), 16);
                    remainder = parseInt(hex.slice(0, hex.length - 8), 16);
                } else {
                    // remainder = BigInt(0);
                    values[i] = remainder;
                    remainder = 0;
                }
            }
            assert.ok(remainder.toString() === '0');
            this.preallocated = values;
            this.highestBit = this.getHighestBit();
        }

        return this;
    }

    __sub__(other: JUCEBigInteger): JUCEBigInteger {
        return this.copy().__isub__(other);
    }
    __isub__(other: JUCEBigInteger): this {
        if (other.isNegative()) {
            other.negative = false;
            // this += other;
            // return this;
            return this.__iadd__(other);
        }
        if (this.isNegative()) {
            this.negate();
            // this += other;
            this.__iadd__(other);
            this.negate();
            return this;
        }
        if (this.compareAbsolute(other) < 0) {
            const temp = other.copy();
            this.swapWith(temp);
            // this -= temp;
            this.__isub__(temp);
            this.negate();
            return this;
        }
        const numInts = sizeNeededToHold(this.getHighestBit());
        const maxOtherInts = sizeNeededToHold(other.getHighestBit());
        assert.ok(numInts >= maxOtherInts);
        const values = this.getValues();
        const otherValues = other.getValues();
        let amountToSubtract = 0;

        for (let i = 0; i < numInts; i++) {
            if (i < maxOtherInts) {
                amountToSubtract += otherValues[i];
            }
            if (values[i] >= amountToSubtract) {
                values[i] = (values[i] - amountToSubtract);
                amountToSubtract = 0;
            } else {
                const n = (values[i] + 4294967296) - amountToSubtract;
                assert.ok(n >= 0 && n <= UINT_MAX);
                values[i] = n;

                amountToSubtract = 1;
            }
        }
        this.preallocated = values;
        this.highestBit = this.getHighestBit();
        return this;
    }

    __mul__(other: JUCEBigInteger) {
        return this.copy().__imul__(other);
    }
    __imul__(other: JUCEBigInteger) {
        let n = this.getHighestBit();
        let t = other.getHighestBit();

        const wasNegative = this.isNegative();
        this.setNegative(false);

        const total = new JUCEBigInteger();
        total.highestBit = n + t + 1;
        const totalValues = total.ensureSize(sizeNeededToHold(total.highestBit) + 1);

        n >>= 5;
        t >>= 5;

        const m = other.copy();
        m.setNegative(false);

        const mValues = m.getValues();
        const values = this.getValues();

        // for (i in range(t+1)) {
        for (let i = 0; i < t+1; i++) {
            // uint32 c = 0
            let c = 0;
            for (let j = 0; j < n+1; j++) {
            // for (j in range(n+1)) {
                // uv = (uint64) totalValues[i + j] + (uint64) values[j] * (uint64) mValues[i] + (uint64) c
                // const uv = totalValues[i + j] + values[j] * mValues[i] + c;
                const uv = BigInt(totalValues[i + j]) + BigInt(values[j]) * BigInt(mValues[i]) + BigInt(c);
                // totalValues[i + j] = (uint32) uv
                totalValues[i + j] = Number(uv & BigInt(0xffffffff));
                // c = static_cast<uint32>(uv >> 32)
                // c = uv >> 32;
                if (uv > UINT_MAX) {
                    const hex = uv.toString(16);
                    c = parseInt(hex.slice(0, hex.length - 8), 16);
                } else {
                    c = 0;
                }
                assert.ok(c >= 0 && c <= 0xffffffff);
            }
            totalValues[i + n + 1] = c;
        }
        total.highestBit = total.getHighestBit();
        // total.setNegative(wasNegative ^ other.isNegative());
        total.setNegative(other.isNegative() !== wasNegative);
        this.swapWith(total);

        return this;
    }

    __truediv__(other: JUCEBigInteger): JUCEBigInteger {
        const a = this.copy();
        a.divideBy(other, new JUCEBigInteger());
        return a;
    }
    __itruediv__(other: JUCEBigInteger): this {
        this.divideBy(other, new JUCEBigInteger());
        return this;
    }
    __mod__(divisor: JUCEBigInteger): JUCEBigInteger {
        return this.copy().__imod__(divisor);
    }
    __imod__(divisor: JUCEBigInteger): this {
        const remainder = new JUCEBigInteger();
        this.divideBy(divisor, remainder);
        this.swapWith(remainder);
        return this;
    }
    __lshift__(bits: number): JUCEBigInteger {
        return this.copy().__ilshift__(bits);
    }
    __ilshift__(bits: number): this {
        this.shiftBits(bits, 0);
        return this;
    }
    __rshift__(bits: number): JUCEBigInteger {
        return this.copy().__irshift__(bits);
    }
    __irshift__(bits: number): this {
        this.shiftBits(-bits, 0);
        return this;
    }
    __getitem__(bit: number): boolean {
        return bit <= this.highestBit && bit >= 0 && ((this.getValues()[bitToIndex(bit)] & bitToMask(bit)) != 0);
    }
    getValues(): Uint32Array { return this.preallocated; }
    isNegative(): boolean { return this.negative && !this.isZero(); }
    setNegative(v: boolean): void { this.negative = v; }
    isZero(): boolean { return this.getHighestBit() < 0; }
    isOne(): boolean { return this.getHighestBit() == 0 && !this.negative; }
    negate(): void { this.negative = !this.negative && !this.isZero(); }

    clear(): void {
        const i = new JUCEBigInteger();
        this.swapWith(i);
    }
    compare(other: JUCEBigInteger): number {
        const isNeg = this.isNegative();
        if (isNeg == other.isNegative()) {
            const absComp = this.compareAbsolute(other);
            if (isNeg) {
                return -absComp;
            }
            return absComp;
        }
        if (isNeg) {
            return -1;
        }
        return 1;
    }
    compareAbsolute(other: JUCEBigInteger): number {
        const h1 = this.getHighestBit();
        const h2 = other.getHighestBit();

        if (h1 > h2) return 1;
        if (h1 < h2) return -1;

        const values = this.getValues();
        const otherValues = other.getValues();
        // for (i in range(bitToIndex(h1), -1, -1) {
        for (let i = bitToIndex(h1); i >= 0; i--) {
            if (values[i] != otherValues[i]) {
                return values[i] > otherValues[i] ? 1 : -1;
            }
        }
        return 0;
    }
    ensureSize(numVals: number): Uint32Array {
        const diff = numVals - this.preallocated.length;
        if (diff > 0) {
            this.preallocated = new Uint32Array([...this.preallocated, ...(new Uint32Array(diff))]);
            // this.preallocated.extend([0 for (i in range(diff)]);
        }
        return this.getValues();
    }
    copyFrom(other: JUCEBigInteger) {
        this.highestBit = Number(other.highestBit);
        this.negative = Boolean(other.negative);
        // this.preallocated = [int(v) for (v in other.preallocated];
        this.preallocated = new Uint32Array([...other.preallocated]);
    }
    copy(): JUCEBigInteger {
        const _copy = new JUCEBigInteger();
        _copy.copyFrom(this);
        return _copy;
    }
    swapWith(other: JUCEBigInteger) {
        const temp = this.copy();
        this.copyFrom(other);
        other.copyFrom(temp);
    }

    getHighestBit(): number {
        const values = this.getValues();
        // for (i in range(bitToIndex(this.highestBit), -1, -1) {
        for (let i = bitToIndex(this.highestBit); i >= 0; i--) {
            const n = values[i];
            if (n) {
                return findHighestSetBit(n) + (i << 5);
            }
        }
        return -1;
    }

    findNextSetBit(j: number): number {
        const values = this.getValues();
        // for (i in range(j, highestBit + 1)) {
        for (let i = j; i <  this.highestBit + 1; i++) {
            if ((values[bitToIndex(i)] & bitToMask(i)) != 0) {
                return i;
            }
        }
        return -1;
    }

    divideBy(divisor: JUCEBigInteger, remainder: JUCEBigInteger): JUCEBigInteger {
        // Returns the remainder
        const divHB = divisor.getHighestBit();
        const ourHB = this.getHighestBit();
        if (divHB < 0 || ourHB < 0) {
            // division by zero
            remainder.clear();
            this.clear();
        } else {
            const wasNegative = this.isNegative();

            this.swapWith(remainder);
            remainder.setNegative(false);
            this.clear();
            const temp = divisor.copy();
            temp.setNegative(false);

            let leftShift = ourHB - divHB;
            // temp <<= leftShift;
            temp.__ilshift__(leftShift);

            while (leftShift >= 0) {
                if (remainder.compareAbsolute(temp) >= 0) {
                    // remainder -= temp;
                    remainder.__isub__(temp);
                    this.setBit1(leftShift);
                }
                leftShift -= 1;
                if (leftShift >= 0) {
                    // temp >>= 1;
                    temp.__irshift__(1);
                }
            }
            // this.negative = wasNegative ^ divisor.isNegative();
            this.negative = divisor.isNegative() !== wasNegative;
            remainder.setNegative(wasNegative);
        }
        return remainder;
    }

    exponentModulo(exponent: JUCEBigInteger, modulus: JUCEBigInteger) {
        this.__imod__(modulus);
        const exp_ = exponent.copy();
        exp_.__imod__(modulus);

        if (modulus.getHighestBit() <= 32 || modulus.__mod__(create32bit(2)).__int__() == 0) {
            const a = this.copy();
            const n = exp_.getHighestBit();

            // for (i in range(n - 1, -1, -1) {
            for (let i = n -1; i >= 0; i--) {
                this.__imul__(this.copy());

                if (exp_.__getitem__(i)) {
                    this.__imul__(a);
                }

                if (this.compareAbsolute(modulus) >= 0) {
                    this.__imod__(modulus);
                }
            }
        } else {
            const Rfactor = modulus.getHighestBit() + 1;
            const R = create32bit(1);
            R.shiftLeft(Rfactor, 0);

            let R1 = new JUCEBigInteger();
            let m1 = new JUCEBigInteger();
            const g = new JUCEBigInteger();
            const { x, y } = g.extendedEuclidean(modulus, R, m1, R1);
            m1 = x;
            R1 = y;

            if (!g.isOne()) {
                const a = this.copy();

                for (let i = exp_.getHighestBit() - 1; i >= 0; i--) {
                    this.__imul__(this.copy());

                    if (exp_.__getitem__(i)) {
                        this.__imul__(a);
                    }

                    if (this.compareAbsolute(modulus) >= 0) {
                        this.__imod__(modulus);
                    }
                }
            } else {
                const t = this.__mul__(R);
                const am = t.__mod__(modulus);
                const xm = am.copy();
                const um = R.__mod__(modulus);
                // for (i in range(exp_.getHighestBit() - 1, -1, -1)) {
                for (let i = exp_.getHighestBit() - 1; i >= 0; i--) {
                    xm.montgomeryMultiplication(xm, modulus, m1, Rfactor);

                    if (exp_.__getitem__(i)) {
                        xm.montgomeryMultiplication(am, modulus, m1, Rfactor);
                    }
                }
                xm.montgomeryMultiplication(create32bit(1), modulus, m1, Rfactor),
                this.swapWith(xm);
            }
        }
    }

    montgomeryMultiplication(other: JUCEBigInteger, modulus: JUCEBigInteger, modulusp: JUCEBigInteger, k: number) {
        this.__imul__(other);
        const t = this.copy();

        this.setRange(k, this.highestBit - k + 1, false);
        this.__imul__(modulusp);

        this.setRange(k, this.highestBit - k + 1, false);
        this.__imul__(modulus);
        this.__iadd__(t);
        this.shiftRight(k, 0);

        if (this.compare(modulus) >= 0) {
            this.__isub__(modulus);
        } else if (this.isNegative()) {
            this.__iadd__(modulus);
        }
    }

    extendedEuclidean(a: JUCEBigInteger, b: JUCEBigInteger, x: JUCEBigInteger, y: JUCEBigInteger) {
        let p = a.copy();
        let q = b.copy();
        let gcd = create32bit(1);
        const tempValues: Array<JUCEBigInteger> = [];

        while (!q.isZero()) {
            // tempValues.push(p / q);
            tempValues.push(p.__truediv__(q));
            gcd.copyFrom(q);
            // q = p % q;
            q.copyFrom(p.__mod__(q));
            p.copyFrom(gcd);
        }
        x.clear();
        y = create32bit(1);

        // for (i in range(1, tempValues.length) {
        for (let i = 1; i <  tempValues.length; i++) {
            const v = tempValues[tempValues.length - i - 1];

            if ((i & 1) != 0) {
                // x += y * v;
                x.__iadd__(y.__mul__(v));
            } else {
                // y += x * v;
                y.__iadd__(x.__mul__(v));
            }
        }
        // if (gcd.compareAbsolute(y * b - x * a) != 0)) {
        if (gcd.compareAbsolute(y.__mul__(b).__sub__(x.__mul__(a))) != 0) {
            x.negate();
            x.swapWith(y);
            x.negate();
        }
        this.swapWith(gcd);
        return {x, y};
    }


    setRange(startBit: number, numBits: number, shouldBeSet: boolean): this {
        // for (i in range(numBits - 1, -1, -1)) {
        for (let i = numBits - 1; i >= 0; i--) {
            this.setBit2(startBit, shouldBeSet);
            startBit ++;
        }
        return this;
    }

    getBitRangeAsInt(startBit: number, numBits: number): number {
        if (numBits > 32) {
            // use getBitRange() if you need more than 32 bits..
            assert.ok(false);
            numBits = 32;
        }
        numBits = Math.min(numBits, this.highestBit + 1 - startBit);

        if (numBits <= 0) {
            return 0;
        }
        const pos = bitToIndex(startBit);
        const offset = startBit & 31;
        const endSpace = 32 - numBits;
        const values = this.getValues();

        let n = values[pos] >>> offset;

        if (offset > endSpace) {
            n |= (values[pos + 1] << (32 - offset));
        }

        // return n & (0xffffffff >> endSpace);
        const v = n & (0xffffffff >>> endSpace);
        return v;
    }

    setBitRangeAsInt(startBit: number, numBits: number, valueToSet: number): this {
        if (numBits > 32) {
            assert.ok(false);
            numBits = 32;
        }
        for (let i = 0; i < numBits; i++) {
            this.setBit2(startBit + i, (valueToSet & 1) != 0);
            valueToSet >>= 1;
        }
        return this;
    }

    setBit1(bit: number): this {
        if (bit >= 0) {
            if (bit > this.highestBit) {
                this.ensureSize(sizeNeededToHold(bit));
                this.highestBit = bit;
            }
            this.preallocated[bitToIndex(bit)] |= bitToMask(bit);
        }
        return this;
    }

    setBit2(bit: number, shouldBeSet: boolean): this {
        if (shouldBeSet) {
            this.setBit1(bit);
        } else {
            this.clearBit(bit);
        }
        return this;
    }

    clearBit(bit: number) {
        if (bit >= 0 && bit <= this.highestBit) {
            this.preallocated[bitToIndex(bit)] &= ~bitToMask(bit);

            if (bit == this.highestBit) {
                this.highestBit = this.getHighestBit();
            }
        }
        return this;
    }

    shiftBits(bits: number, startBit: number) {
        if (this.highestBit >= 0) {
            if (bits < 0) {
                this.shiftRight(-bits, startBit);
            } else if (bits > 0) {
                this.shiftLeft(bits, startBit);
            }
        }
        return this;
    }

    shiftRight(bits: number, startBit: number): void {
        assert.ok(bits >= 0);
        if (startBit > 0) {
            // for (i in range(startBit, this.highestBit + 1)) {
            for (let i = startBit; i < this.highestBit + 1; i++) {
                this.setBit2(i, Boolean(this.preallocated[i + bits]));
            }
            this.highestBit = this.getHighestBit();
        } else {
            if (bits > this.highestBit) {
                this.clear();
            } else {
                const wordsToMove = bitToIndex(bits);
                let top = 1 + bitToIndex(this.highestBit) - wordsToMove;
                this.highestBit -= bits;
                const values = this.getValues();

                if (wordsToMove > 0) {
                    for (let i = 0; i < top; i++) {
                        values[i] = values[i + wordsToMove];
                    }
                    for (let i = 0; i < wordsToMove; i++) {
                        values[top + i] = 0;
                    }
                    bits &= 31;
                }
                if (bits != 0) {
                    const invBits = 32 - bits;
                    top--;

                    for (let i = 0; i < top; i++) {
                        values[i] = (values[i] >>> bits) | (values[i + 1] << invBits);
                        values[i] &= 0xffffffff;
                    }
                    values[top] = (values[top] >>> bits);
                }

                this.preallocated = values;
                this.highestBit = this.getHighestBit();
            }
        }
    }

    shiftLeft(bits: number, startBit: number): void {
        assert.ok(bits >= 0);
        this.highestBit = this.getHighestBit();
        if (startBit > 0) {
            for (let i = this.highestBit; i >= startBit; i--) {
                this.setBit2(i + bits, this.__getitem__(i));
            }
            // while (;--bits >= 0;) {
            for (let i = bits - 1; i >= 0; i--) {
                this.clearBit(i + startBit);
            }
        } else {
            const values = this.ensureSize(sizeNeededToHold(this.highestBit + bits));
            const wordsToMove = bitToIndex(bits);
            const numOriginalInts = bitToIndex(this.highestBit);
            this.highestBit += bits;

            if (wordsToMove > 0) {
                for (let i = numOriginalInts; i >= 0; i--) {
                    values[i + wordsToMove] = values[i];
                }
                for (let i = 0; i < wordsToMove; i++) {
                    values[i] = 0;
                }
                bits &= 31;
            }
            if (bits != 0) {
                const invBits = 32 - bits;

                for (let i = bitToIndex (this.highestBit); i > wordsToMove; i--) {
                    values[i] = (values[i] << bits) | (values[i - 1] >>> invBits);
                }

                values[wordsToMove] = values[wordsToMove] << bits;
            }
            this.highestBit = this.getHighestBit();
        }
    }

    toString(base = 10): string {
        let s = '';
        const v = this.copy();
        if ([2, 8, 16].includes(base)) {
            // const bits = 1 if (base == 2 else (3 if (base == 8 else 4);
            const bits = base == 2 ? 1 : (base == 8 ? 3 : 4);
            const hexDigits = '0123456789abcdef';

            while (true) {
                const remainder = v.getBitRangeAsInt(0, bits);
                // v >>= bits;
                v.__irshift__(bits);

                if (remainder == 0 && v.isZero()) {
                    break;
                }
                s = hexDigits[remainder] + s;
            }
        } else if (base == 10) {
            // const ten = create32bit(10);
            // let remainder = new JUCEBigInteger();

            // while (true) {
            //     remainder = v.divideBy(ten, remainder);

            //     if (remainder.isZero() && v.isZero()) {
            //         break;
            //     }
            //     s = remainder.getBitRangeAsInt(0, 8).toString() + s;
            // }
            // This is faster
            let hex = '';
            const charBuffer = this.toMemoryBlock();
            charBuffer.forEach(v => {
                const a = v.toString(16);
                hex = (v < 0x10 ? '0' + a : a) + hex;
            });
            const d = BigInt('0x' + hex);
            s = d.toString();
        } else {
            assert.ok(false);
            // can't do the specified base!
            return '';
        }
        assert.ok(s.length);

        return this.isNegative() ? ('-' + s) : s;
    }

    parseString(text: string, base: number) {
        assert.ok(text.length > 0);
        this.clear();

        this.setNegative(text[0] === '-');

        if ([2, 8, 16].includes(base)) {
            const bits = base == 2 ? 1 : (base == 8 ? 3 : 4);

            for (const t of text) {
                // c = ord(t);
                const c = t.charCodeAt(0)
                const digit = getHexDigitValue(c);

                if (digit < base) {
                    // this <<= bits;
                    this.__ilshift__(bits);
                    // this += create32bit(digit);
                    this.__iadd__(create32bit(digit));
                } else if (c == 0) {
                    break;
                }
            }
        } else if (base === 10) {
            const ten = create32bit(10);

            for (const t of text) {
                // c = ord(t);
                const c = t.charCodeAt(0)

                if (c >= '0'.charCodeAt(0) && c <= '9'.charCodeAt(0)) {
                    // this *= ten;
                    this.__imul__(ten);
                    // this += create32bit(c - '0'.charCodeAt(0));
                    this.__iadd__(create32bit(c - '0'.charCodeAt(0)));
                } else if (c === 0) {
                    break;
                }
            }
        } else {
            assert.ok(false);
            // can't do the specified base!
        }
    }

    toMemoryBlock(): Uint8Array {
        let i = this.preallocated.length - 1;
        for (; i >= 1; i--) {
            if (this.preallocated[i] !== 0) {
                break;
            }
        }


        const cropped = new Uint32Array(this.preallocated.slice(0, i + 1));
        return new Uint8Array(cropped.buffer);;
    }

    loadFromMemoryBlock(data: string) {
        const numBytes = data.length;
        const numInts = 1 + Math.floor(numBytes / 4);
        const values = this.ensureSize(numInts);

        for (let i = 0; i < numInts - 1; i++) {
            const startIdx = i * 4;
            const endIdx = startIdx + 4;
            const part = data.slice(startIdx, endIdx);
            // works. probably simpler method
            const val = new Uint32Array(new Uint8Array(Buffer.from(part)).buffer)[0];
            values[i] = val
        }
        values[numInts - 1] = 0;
        this.preallocated = values;

        for (let i = numBytes & 0xfffffffc; i < numBytes; i++) {
            this.setBitRangeAsInt(i << 3, 8, data[i].charCodeAt(0));
        }
        this.highestBit = numBytes * 8;
        this.highestBit = this.getHighestBit();
    }
};



 class RSAKey {
    part1: JUCEBigInteger;
    part2: JUCEBigInteger;

    constructor() {
        this.part1 = new JUCEBigInteger();
        this.part2 = new JUCEBigInteger();
    }

    static createFromKeystring(keystring: string): RSAKey {
        const parts = keystring.split(',');
        assert.ok(parts.length == 2);
        const key = new RSAKey();

        key.part1.parseString(parts[0], 16);
        key.part2.parseString(parts[1], 16);

        return key;
    }

    applyToValue(value: JUCEBigInteger): JUCEBigInteger {
        assert.ok(!this.part1.isZero());
        assert.ok(!this.part2.isZero());
        let result = new JUCEBigInteger();

        while (!value.isZero()) {
            // result *= this.part2.copy();
            result.__imul__(this.part2);

            let remainder = new JUCEBigInteger();
            remainder = value.divideBy(this.part2, remainder);

            remainder.exponentModulo(this.part1, this.part2);

            // result += remainder;
            result.__iadd__(remainder);
        }

        value.swapWith(result);
        return value;
    }
};



const message = 'Super secret message!'

// eg. keypair created in JUCE, 512 bits
// juce::RSAKey pub, priv;
// juce::RSAKey::createKeyPair(pub, priv, 512);
const juce_rsa_pub = '11,5e77dd9642dde73c270a5583be086d8cc67eeb585e3bd5029a3f0d73d6148abb5aeecb0ae4d29c57b283cf91ebd0d22dcfa9aecc5b9684a0927b03083358fba1'
const juce_rsa_priv = '535ab475864b538f6dbdd2fbc5cb337c36ac3911bc8f255ca637a275446c7a67fcc31fb3cefc1dcb0464c55e7721dc260700cfeb3c9e1280e9d2fb946fa2bbf1,5e77dd9642dde73c270a5583be086d8cc67eeb585e3bd5029a3f0d73d6148abb5aeecb0ae4d29c57b283cf91ebd0d22dcfa9aecc5b9684a0927b03083358fba1'

// using the JUCE BigInteger and RSAKey classes, encrypting/decrypting
// the above message yields these results

// juce::MemoryOutputStream text;
// juce::BigInteger val1;
// text << message;
// val1.loadFromMemoryBlock(text.getMemoryBlock());
// val1.toString(10))
const juce_message_number = '48808467565706048178356521455376678508857750746451'
// # priv.applyToValue(val1);
// # auto encryptedMessage = val1.toString(16);
const juce_encrypted_message = '550c52c8b504e105cccb7d9f16021ba8eecb32c16da6c72f3069dafecf6116dbb54f7f6225cd3dcd219dc5f919ebb7e1e904e14ed1a041a27c5715edeaed88f5'
// # juce::BigInteger val2;
// # val2.parseString(encryptedMessage, 16);
// # pub.applyToValue(val2);
// # auto decryptedMessage = val2.toMemoryBlock().toString();
const juce_decrypted_message = 'Super secret message!'

// # Test old method

console.log('Test JUCE method');
console.log('loading keys...');
const pub = RSAKey.createFromKeystring(juce_rsa_pub);
const priv = RSAKey.createFromKeystring(juce_rsa_priv);

assert.ok(pub.part1.__ne__(priv.part1));
assert.ok(pub.part2.__eq__(priv.part2));

console.log('loading message...');
let val1 = new JUCEBigInteger();
val1.loadFromMemoryBlock(message);

console.log('comparing loaded message with JUCE');
assert.ok(val1.toString(10) == juce_message_number);

console.log('encrypting message');
val1 = priv.applyToValue(val1);
const encrypted_message_16 = val1.toString(16);

console.log('comparing encrypted message with JUCE');
assert.ok(encrypted_message_16 == juce_encrypted_message);

let val2 = new JUCEBigInteger();
val2.parseString(encrypted_message_16, 16);

console.log('decrypting message');
val2 = pub.applyToValue(val2);
console.log('comparing decrypted message with JUCE');
let mb = val2.toMemoryBlock();
mb = mb.filter(v => v !== 0);
const decrypted_message = Buffer.from(mb).toString('utf8').trim();

assert.ok(decrypted_message.trim() == message.trim());
console.log('success!');
