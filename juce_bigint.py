from typing import Type, List, Tuple


class BigInteger:
    pass


def countNumberOfBits(n: int):
    n -= ((n >> 1) & 0x55555555)
    n = (((n >> 2) & 0x33333333) + (n & 0x33333333))
    n = (((n >> 4) + n) & 0x0f0f0f0f)
    n += (n >> 8)
    n += (n >> 16)
    return n & 0x3f


def findHighestSetBit(n: int):
    assert n != 0
    n |= (n >> 1)
    n |= (n >> 2)
    n |= (n >> 4)
    n |= (n >> 8)
    n |= (n >> 16)
    return countNumberOfBits(n >> 1)


def bitToMask(bit: int):
    return 1 << (bit & 31)


def bitToIndex(bit: int):
    return bit >> 5


def sizeNeededToHold(highestBit: int):
    return (highestBit >> 5) + 1


def getHexDigitValue(digit: int):
    d = digit - ord('0')

    if d < 10:
        return d

    d += ord('0') - ord('a')

    if d < 6:
        return d + 10

    d += ord('a') - ord('A')

    if d < 6:
        return d + 10

    return -1


def create32bit(num: int):
    assert num >= 0
    assert num < (1 << 32)
    i = BigInteger()
    i.highestBit = 31
    i.negative = num < 0
    i.preallocated = [num]
    i.highestBit = i.getHighestBit()
    return i


class BigInteger:
    __slots__ = ('highestBit', 'negative', 'preallocated')

    def __init__(self):
        self.highestBit = -1
        self.negative = False
        self.preallocated = [0]

    def __repr__(self):
        return str(self.__int__())

    def __int__(self):
        a = bytearray()
        for v in self.getValues():
            a.extend(v.to_bytes(4, 'little'))
        int_val = int.from_bytes(a, "little")
        if self.isNegative():
            int_val = -int_val
        return int_val

    def __eq__(self, other: Type[BigInteger]):
        return self.compare(other) == 0

    def __ne__(self, other: Type[BigInteger]):
        return self.compare(other) != 0

    def __lt__(self, other: Type[BigInteger]):
        return self.compare(other) < 0

    def __le__(self, other: Type[BigInteger]):
        return self.compare(other) <= 0

    def __gt__(self, other: Type[BigInteger]):
        return self.compare(other) > 0

    def __ge__(self, other: Type[BigInteger]):
        return self.compare(other) >= 0

    def __neg__(self):
        result = self.copy()
        result.setNegative(not result.isNegative())
        return result

    def __add__(self, other: Type[BigInteger]):
        return self.copy().__iadd__(other)

    def __iadd__(self, other: Type[BigInteger]):
        if (other.isNegative()):
            self -= -other
            return self

        if (self.isNegative()):
            if (self.compareAbsolute(other) < 0):
                temp = self.copy()
                temp.negate()
                self.copyFrom(other)
                self -= temp
            else:
                self.negate()
                self -= other
                self.negate()
        else:
            self.highestBit = max(self.highestBit, other.highestBit) + 1

            numInts = sizeNeededToHold(self.highestBit)
            values = self.ensureSize(numInts)
            otherValues = other.getValues()

            remainder = 0

            for i in range(numInts):
                remainder += values[i]

                if i < len(otherValues):
                    remainder += otherValues[i]

                values[i] = remainder
                values[i] &= 0xffffffff

                remainder >>= 32

            assert remainder == 0
            self.preallocated = values
            self.highestBit = self.getHighestBit()

        return self

    def __sub__(self, other: Type[BigInteger]):
        return self.copy().__isub__(other)

    def __isub__(self, other: Type[BigInteger]):
        if (other.isNegative()):
            other.negative = False
            self += other
            return self

        if (self.isNegative()):
            self.negate()
            self += other
            self.negate()
            return self

        if (self.compareAbsolute(other) < 0):
            temp = other.copy()
            self.swapWith(temp)
            self -= temp
            self.negate()
            return self

        numInts = sizeNeededToHold(self.getHighestBit())
        maxOtherInts = sizeNeededToHold(other.getHighestBit())
        assert numInts >= maxOtherInts
        values = self.getValues()
        otherValues = other.getValues()
        amountToSubtract = 0

        for i in range(numInts):
            if i < maxOtherInts:
                amountToSubtract += otherValues[i]

            if (values[i] >= amountToSubtract):
                values[i] = (values[i] - amountToSubtract)
                amountToSubtract = 0
            else:
                n = (values[i] + (1 << 32)) - amountToSubtract
                values[i] = n

                amountToSubtract = 1

        self.preallocated = values
        self.highestBit = self.getHighestBit()
        return self

    def __mul__(self, other: Type[BigInteger]):
        return self.copy().__imul__(other)

    def __imul__(self, other: Type[BigInteger]):
        n = self.getHighestBit()
        t = other.getHighestBit()

        wasNegative = self.isNegative()
        self.setNegative(False)

        total = BigInteger()
        total.highestBit = n + t + 1
        totalValues = total.ensureSize(sizeNeededToHold(total.highestBit) + 1)

        n >>= 5
        t >>= 5

        m = other.copy()
        m.setNegative(False)

        mValues = m.getValues()
        values = self.getValues()

        for i in range(t+1):
            # uint32 c = 0
            c = 0

            for j in range(n+1):
                # uv = (uint64) totalValues[i + j] + (uint64) values[j] * (uint64) mValues[i] + (uint64) c
                uv = totalValues[i + j] + values[j] * mValues[i] + c
                # totalValues[i + j] = (uint32) uv
                totalValues[i + j] = (uv & 0xffffffff)
                # c = static_cast<uint32>(uv >> 32)
                c = uv >> 32
                assert c >= 0 and c <= 0xffffffff

            totalValues[i + n + 1] = c

        total.highestBit = total.getHighestBit()
        total.setNegative(wasNegative ^ other.isNegative())
        self.swapWith(total)

        return self

    def __truediv__(self, other: Type[BigInteger]) -> Type[BigInteger]:
        a = self.copy()
        a.divideBy(other, BigInteger())
        return a

    def __itruediv__(self, other: Type[BigInteger]) -> Type[BigInteger]:
        self.divideBy(other, BigInteger())
        return self

    def __mod__(self, divisor: Type[BigInteger]):
        return self.copy().__imod__(divisor)

    def __imod__(self, divisor: Type[BigInteger]):
        remainder = BigInteger()
        self.divideBy(divisor, remainder)
        self.swapWith(remainder)
        return self

    def __lshift__(self, bits: int):
        return self.copy().__ilshift__(bits)

    def __ilshift__(self, bits: int):
        self.shiftBits(bits, 0)
        return self

    def __rshift__(self, bits: int):
        return self.copy().__irshift__(bits)

    def __irshift__(self, bits: int):
        self.shiftBits(-bits, 0)
        return self

    def __getitem__(self, bit: int) -> bool:
        return bit <= self.highestBit and bit >= 0 and ((self.getValues()[bitToIndex(bit)] & bitToMask(bit)) != 0)

    def getValues(self): return self.preallocated
    def isNegative(self) -> bool: return self.negative and not self.isZero()
    def setNegative(self, v: bool): self.negative = v
    def isZero(self) -> bool: return self.getHighestBit() < 0
    def isOne(self) -> bool: return self.getHighestBit() == 0 and not self.negative
    def negate(self): self.negative = (not self.negative) and not self.isZero()

    def clear(self):
        i = BigInteger()
        self.swapWith(i)

    def compare(self, other: Type[BigInteger]) -> int:
        isNeg = self.isNegative()

        if (isNeg == other.isNegative()):
            absComp = self.compareAbsolute(other)
            if isNeg:
                return -absComp
            else:
                return absComp

        if isNeg:
            return -1
        else:
            return 1

    def compareAbsolute(self, other: Type[BigInteger]) -> int:
        h1 = self.getHighestBit()
        h2 = other.getHighestBit()

        if h1 > h2:
            return 1
        if h1 < h2:
            return -1

        values = self.getValues()
        otherValues = other.getValues()

        for i in range(bitToIndex(h1), -1, -1):
            if values[i] != otherValues[i]:
                if values[i] > otherValues[i]:
                    return 1
                else:
                    return -1

        return 0

    def ensureSize(self, numVals: int) -> List[int]:
        diff = numVals - len(self.preallocated)
        if diff > 0:
            self.preallocated.extend([0 for i in range(diff)])

        return self.getValues()

    def copyFrom(self, other: Type[BigInteger]):
        self.highestBit = int(other.highestBit)
        self.negative = bool(other.negative)
        self.preallocated = [int(v) for v in other.preallocated]

    def copy(self) -> Type[BigInteger]:
        new = BigInteger()
        new.copyFrom(self)
        return new

    def swapWith(self, other: Type[BigInteger]):
        temp = self.copy()
        self.copyFrom(other)
        other.copyFrom(temp)

    def getHighestBit(self) -> int:
        values = self.getValues()

        for i in range(bitToIndex(self.highestBit), -1, -1):
            n = values[i]
            if (n):
                return findHighestSetBit(n) + (i << 5)

        return -1

    def findNextSetBit(self, j: int) -> int:
        values = getValues()
        for i in range(j, self.highestBit + 1):
            if ((values[bitToIndex(i)] & bitToMask(i)) != 0):
                return i

        return -1

    def divideBy(self, divisor: Type[BigInteger], remainder: Type[BigInteger]):
        """Returns the remainder"""
        divHB = divisor.getHighestBit()
        ourHB = self.getHighestBit()

        if (divHB < 0 or ourHB < 0):
            # division by zero
            remainder.clear()
            self.clear()
        else:
            wasNegative = self.isNegative()

            self.swapWith(remainder)
            remainder.setNegative(False)
            self.clear()

            temp = divisor.copy()
            temp.setNegative(False)

            leftShift = ourHB - divHB
            temp <<= leftShift

            while (leftShift >= 0):
                if (remainder.compareAbsolute(temp) >= 0):
                    remainder -= temp
                    self.setBit1(leftShift)

                leftShift -= 1
                if (leftShift >= 0):
                    temp >>= 1

            self.negative = wasNegative ^ divisor.isNegative()
            remainder.setNegative(wasNegative)
        return remainder

    def exponentModulo(self, exponent: Type[BigInteger], modulus: Type[BigInteger]):
        self %= modulus
        exp_ = exponent.copy()
        exp_ %= modulus

        if (modulus.getHighestBit() <= 32 or int(modulus) % 2 == 0):
            a = self.copy()
            n = exp_.getHighestBit()

            for i in range(n - 1, -1, -1):
                self *= self.copy()

                if (exp_[i]):
                    self *= a

                if (self.compareAbsolute(modulus) >= 0):
                    self %= modulus
        else:
            Rfactor = modulus.getHighestBit() + 1
            R = create32bit(1)
            R.shiftLeft(Rfactor, 0)

            R1 = BigInteger()
            m1 = BigInteger()
            g = BigInteger()
            m1, R1 = g.extendedEuclidean(modulus, R, m1, R1)

            if not g.isOne():
                a = self.copy()

                for i in range(exp_.getHighestBit() - 1, -1, -1):
                    self *= self.copy()

                    if (exp_[i]):
                        self *= a

                    if (self.compareAbsolute(modulus) >= 0):
                        self %= modulus
            else:
                t = (self * R)
                am = t % modulus
                xm = am.copy()
                um = R % modulus

                for i in range(exp_.getHighestBit() - 1, -1, -1):
                    xm.montgomeryMultiplication(xm, modulus, m1, Rfactor)

                    if (exp_[i]):
                        xm.montgomeryMultiplication(am, modulus, m1, Rfactor)

                xm.montgomeryMultiplication(
                    create32bit(1), modulus, m1, Rfactor)
                self.swapWith(xm)

    def montgomeryMultiplication(self, other: Type[BigInteger], modulus: Type[BigInteger], modulusp: Type[BigInteger], k: int):
        self *= other
        t = self.copy()

        self.setRange(k, self.highestBit - k + 1, False)
        self *= modulusp

        self.setRange(k, self.highestBit - k + 1, False)
        self *= modulus
        self += t
        self.shiftRight(k, 0)

        if (self.compare(modulus) >= 0):
            self -= modulus
        elif (self.isNegative()):
            self += modulus

    def extendedEuclidean(self, a: Type[BigInteger], b: Type[BigInteger], x: Type[BigInteger], y: Type[BigInteger]):
        """returns (x, y)"""

        p = a.copy()
        q = b.copy()
        gcd = create32bit(1)
        tempValues: List[BigInteger] = []

        while not q.isZero():
            tempValues.append(p / q)
            gcd = q
            q = p % q
            p = gcd

        x.clear()
        y = create32bit(1)

        for i in range(1, len(tempValues)):
            v = tempValues[len(tempValues) - i - 1]

            if (i & 1) != 0:
                x += y * v
            else:
                y += x * v

        if (gcd.compareAbsolute(y * b - x * a) != 0):
            x.negate()
            x.swapWith(y)
            x.negate()

        self.swapWith(gcd)
        return (x, y)

    def setRange(self, startBit: int, numBits: int, shouldBeSet: bool):
        for i in range(numBits - 1, -1, -1):
            self.setBit2(startBit, shouldBeSet)
            startBit += 1

        return self

    def getBitRangeAsInt(self, startBit: int, numBits: int) -> int:
        if (numBits > 32):
            # use getBitRange() if you need more than 32 bits..
            assert False
            numBits = 32

        numBits = min(numBits, self.highestBit + 1 - startBit)

        if (numBits <= 0):
            return 0

        pos = bitToIndex(startBit)
        offset = startBit & 31
        endSpace = 32 - numBits
        values = self.getValues()

        n = values[pos] >> offset

        if (offset > endSpace):
            n |= (values[pos + 1] << (32 - offset))

        return n & (0xffffffff >> endSpace)

    def setBitRangeAsInt(self, startBit: int, numBits: int, valueToSet: int):
        if numBits > 32:
            assert False
            numBits = 32

        for i in range(numBits):
            self.setBit2(startBit + i, (valueToSet & 1) != 0)
            valueToSet >>= 1

        return self

    def setBit1(self, bit: int):
        if bit >= 0:
            if bit > self.highestBit:
                self.ensureSize(sizeNeededToHold(bit))
                self.highestBit = bit

            self.preallocated[bitToIndex(bit)] |= bitToMask(bit)

        return self

    def setBit2(self, bit: int, shouldBeSet: bool):
        if (shouldBeSet):
            self.setBit1(bit)
        else:
            self.clearBit(bit)

        return self

    def clearBit(self, bit: int):
        if (bit >= 0 and bit <= self.highestBit):
            self.preallocated[bitToIndex(bit)] &= ~bitToMask(bit)

            if (bit == self.highestBit):
                self.highestBit = self.getHighestBit()

        return self

    def shiftBits(self, bits: int, startBit: int):
        if (self.highestBit >= 0):
            if bits < 0:
                self.shiftRight(-bits, startBit)
            elif bits > 0:
                self.shiftLeft(bits, startBit)

        return self

    def shiftRight(self, bits: int, startBit: int):
        assert bits >= 0
        if startBit > 0:
            for i in range(startBit, self.highestBit + 1):
                self.setBit2(i, bool(self.preallocated[i + bits]))

            self.highestBit = self.getHighestBit()
        else:
            if (bits > self.highestBit):
                self.clear()
            else:
                wordsToMove = bitToIndex(bits)
                top = 1 + bitToIndex(self.highestBit) - wordsToMove
                self.highestBit -= bits
                values = self.getValues()

                if (wordsToMove > 0):
                    for i in range(top):
                        values[i] = values[i + wordsToMove]

                    for i in range(wordsToMove):
                        values[top + i] = 0

                    bits &= 31

                if bits != 0:
                    invBits = 32 - bits
                    top -= 1

                    for i in range(top):
                        values[i] = (values[i] >> bits) | (
                            values[i + 1] << invBits)
                        values[i] &= 0xffffffff

                    values[top] = (values[top] >> bits)

                self.preallocated = values
                self.highestBit = self.getHighestBit()

    def shiftLeft(self, bits: int, startBit: int):
        assert bits >= 0
        self.highestBit = self.getHighestBit()

        if (startBit > 0):
            for i in range(self.highestBit, startBit - 1, -1):
                self.setBit2(i + bits, self[i])

            # while (--bits >= 0):
            for b in range(bits - 1, -1, -1):
                self.clearBit(b + startBit)
        else:
            values = self.ensureSize(sizeNeededToHold(self.highestBit + bits))
            wordsToMove = bitToIndex(bits)
            numOriginalInts = bitToIndex(self.highestBit)
            self.highestBit += bits

            if (wordsToMove > 0):
                for i in range(numOriginalInts, -1, -1):
                    values[i + wordsToMove] = values[i]

                for i in range(wordsToMove):
                    values[i] = 0

                bits &= 31

            if (bits != 0):
                invBits = 32 - bits

                for i in range(bitToIndex(self.highestBit), wordsToMove, -1):
                    values[i] = (values[i] << bits) | (
                        values[i - 1] >> invBits)
                    values[i] &= 0xffffffff

                values[wordsToMove] = values[wordsToMove] << bits
                values[wordsToMove] &= 0xffffffff

            self.highestBit = self.getHighestBit()

    def toString(self, base: int) -> str:
        s = ''
        v = self.copy()

        if base in (2, 8, 16):
            bits = 1 if base == 2 else (3 if base == 8 else 4)
            hexDigits = "0123456789abcdef"

            while True:
                remainder = v.getBitRangeAsInt(0, bits)
                v >>= bits

                if remainder == 0 and v.isZero():
                    break

                s = hexDigits[remainder] + s
        elif base == 10:
            ten = create32bit(10)
            remainder = BigInteger()

            while True:
                remainder = v.divideBy(ten, remainder)

                if remainder.isZero() and v.isZero():
                    break

                s = str(remainder.getBitRangeAsInt(0, 8)) + s
        else:
            assert False
            # can't do the specified base!
            return ''

        return "-" + s if self.isNegative() else s

    def parseString(self, text: str, base):
        assert len(text) > 0
        self.clear()

        self.setNegative(text[0] == '-')

        if base in (2, 8, 16):
            bits = 1 if base == 2 else (3 if base == 8 else 4)

            for t in text:
                c = ord(t)
                digit = getHexDigitValue(c)

                if digit < base:
                    self <<= bits
                    self += create32bit(digit)
                elif c == 0 or stringIdx == len(text):
                    break
        elif base == 10:
            ten = create32bit(10)

            for t in text:
                c = ord(t)

                if c >= ord('0') and c <= ord('9'):
                    self *= ten
                    self += create32bit(c - ord('0'))
                elif (c == 0):
                    break

    def toMemoryBlock(self) -> bytearray:
        numBytes: int = (self.getHighestBit() + 8) >> 3
        mb = bytearray()
        values = self.getValues()

        for v in self.preallocated:
            mb.extend(v.to_bytes(4, 'little'))

        while len(mb) and mb[-1] == 0:
            mb.pop()

        return mb

    def loadFromMemoryBlock(self, data: str):
        numBytes = len(data)
        numInts = 1 + int(numBytes / 4)
        values = self.ensureSize(numInts)

        # for (int i = 0; i < (int) numInts - 1; ++i)
        # for (int i = 0; i < (int) numInts - 1; ++i)
        for i in range(numInts - 1):
            startIdx = i * 4
            endIdx = startIdx + 4
            part = data[startIdx:endIdx]

            values[i] = int.from_bytes(part.encode('utf8'), 'little')
            # values[i] = (uint32) ByteOrder::littleEndianInt (addBytesToPointer (data.getData(), (size_t) i * sizeof (uint32)));

        values[numInts - 1] = 0
        self.preallocated = values

        for i in range(numBytes & 0xfffffffc, numBytes):
            self.setBitRangeAsInt(i << 3, 8, ord(data[i]))

        self.highestBit = numBytes * 8
        self.highestBit = self.getHighestBit()


class RSAKey:
    __slots__ = ('part1', 'part2')

    def __init__(self):
        self.part1 = BigInteger()
        self.part2 = BigInteger()

    @ staticmethod
    def createFromKeystring(keystring: str):
        parts = keystring.split(',')
        assert len(parts) == 2

        key = RSAKey()

        key.part1.parseString(parts[0], 16)
        key.part2.parseString(parts[1], 16)
        return key

    def applyToValue(self, value: BigInteger) -> BigInteger:
        assert not self.part1.isZero()
        assert not self.part2.isZero()

        result = BigInteger()

        while not value.isZero():
            result *= self.part2.copy()

            remainder = BigInteger()
            remainder = value.divideBy(self.part2.copy(), remainder)

            remainder.exponentModulo(self.part1.copy(), self.part2.copy())

            result += remainder

        value.swapWith(result)
        return value


message = 'Super secret message!'

# eg. keypair created in JUCE, 512 bits
# juce::RSAKey pub, priv;
# juce::RSAKey::createKeyPair(pub, priv, 512);
juce_rsa_pub = '11,5e77dd9642dde73c270a5583be086d8cc67eeb585e3bd5029a3f0d73d6148abb5aeecb0ae4d29c57b283cf91ebd0d22dcfa9aecc5b9684a0927b03083358fba1'
juce_rsa_priv = '535ab475864b538f6dbdd2fbc5cb337c36ac3911bc8f255ca637a275446c7a67fcc31fb3cefc1dcb0464c55e7721dc260700cfeb3c9e1280e9d2fb946fa2bbf1,5e77dd9642dde73c270a5583be086d8cc67eeb585e3bd5029a3f0d73d6148abb5aeecb0ae4d29c57b283cf91ebd0d22dcfa9aecc5b9684a0927b03083358fba1'

# using the JUCE BigInteger and RSAKey classes, encrypting/decrypting
# the above message yields these results

# juce::MemoryOutputStream text;
# juce::BigInteger val1;
# text << message;
# val1.loadFromMemoryBlock(text.getMemoryBlock());
# val1.toString(10))
juce_message_number = 48808467565706048178356521455376678508857750746451
# priv.applyToValue(val1);
# auto encryptedMessage = val1.toString(16);
juce_encrypted_message = '550c52c8b504e105cccb7d9f16021ba8eecb32c16da6c72f3069dafecf6116dbb54f7f6225cd3dcd219dc5f919ebb7e1e904e14ed1a041a27c5715edeaed88f5'
# juce::BigInteger val2;
# val2.parseString(encryptedMessage, 16);
# pub.applyToValue(val2);
# auto decryptedMessage = val2.toMemoryBlock().toString();
juce_decrypted_message = 'Super secret message!'

# Test old method

print('Test JUCE method')
print('loading keys...')
pub = RSAKey.createFromKeystring(juce_rsa_pub)
priv = RSAKey.createFromKeystring(juce_rsa_priv)

assert pub.part1 != priv.part1
assert pub.part2 == priv.part2

print('loading message...')
val1 = BigInteger()
val1.loadFromMemoryBlock(message)

print('comparing loaded message with JUCE')
assert int(val1) == juce_message_number
assert val1.toString(10) == str(juce_message_number)

print('encrypting message')
val1 = priv.applyToValue(val1)
encrypted_message_16 = val1.toString(16)

print('comparing encrypted message with JUCE')
assert encrypted_message_16 == juce_encrypted_message

val2 = BigInteger()
val2.parseString(encrypted_message_16, 16)

print('decrypting message')
val2 = pub.applyToValue(val2)
print('comparing decrypted message with JUCE')
mb = val2.toMemoryBlock()
decrypted_message = mb.decode('utf8')

assert decrypted_message == message
print('success!')
