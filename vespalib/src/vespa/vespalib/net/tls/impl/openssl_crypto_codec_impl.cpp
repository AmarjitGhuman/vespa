// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include "openssl_crypto_codec_impl.h"
#include "openssl_tls_context_impl.h"
#include "direct_buffer_bio.h"
#include <vespa/vespalib/net/tls/crypto_codec.h>
#include <vespa/vespalib/net/tls/crypto_exception.h>
#include <mutex>
#include <vector>
#include <memory>
#include <stdexcept>
#include <openssl/ssl.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/pem.h>

#include <vespa/log/log.h>
LOG_SETUP(".vespalib.net.tls.openssl_crypto_codec_impl");

#if (OPENSSL_VERSION_NUMBER < 0x10000000L)
// < 1.0 requires explicit thread ID callback support.
#  error "Provided OpenSSL version is too darn old, need at least 1.0"
#endif

/*
 * Beware all ye who dare enter, for this is OpenSSL integration territory.
 * Dragons are known to roam the skies. Strange whispers are heard at night
 * in the mist-covered lands where the forest meets the lake. Rumors of a
 * tome that contains best practices and excellent documentation are heard
 * at the local inn, but no one seems to know where it exists, or even if
 * it ever existed. Be it best that people carry on with their lives and
 * pretend to not know of the beasts that lurk beyond where the torch's
 * light fades and turns to all-enveloping darkness.
 */

namespace vespalib::net::tls::impl {

namespace {

bool verify_buf(const char *buf, size_t len) {
    return ((len < INT32_MAX) && ((len == 0) || (buf != nullptr)));
}

const char* ssl_error_to_str(int ssl_error) noexcept {
    // From https://www.openssl.org/docs/manmaster/man3/SSL_get_error.html
    // Our code paths shouldn't trigger most of these, but included for completeness
    switch (ssl_error) {
        case SSL_ERROR_NONE:
            return "SSL_ERROR_NONE";
        case SSL_ERROR_ZERO_RETURN:
            return "SSL_ERROR_ZERO_RETURN";
        case SSL_ERROR_WANT_READ:
            return "SSL_ERROR_WANT_READ";
        case SSL_ERROR_WANT_WRITE:
            return "SSL_ERROR_WANT_WRITE";
        case SSL_ERROR_WANT_CONNECT:
            return "SSL_ERROR_WANT_CONNECT";
        case SSL_ERROR_WANT_ACCEPT:
            return "SSL_ERROR_WANT_ACCEPT";
        case SSL_ERROR_WANT_X509_LOOKUP:
            return "SSL_ERROR_WANT_X509_LOOKUP";
#if (OPENSSL_VERSION_NUMBER >= 0x10100000L)
        case SSL_ERROR_WANT_ASYNC:
            return "SSL_ERROR_WANT_ASYNC";
        case SSL_ERROR_WANT_ASYNC_JOB:
            return "SSL_ERROR_WANT_ASYNC_JOB";
#endif
#if (OPENSSL_VERSION_NUMBER >= 0x10101000L)
        case SSL_ERROR_WANT_CLIENT_HELLO_CB:
            return "SSL_ERROR_WANT_CLIENT_HELLO_CB";
#endif
        case SSL_ERROR_SYSCALL:
            return "SSL_ERROR_SYSCALL";
        case SSL_ERROR_SSL:
            return "SSL_ERROR_SSL";
        default:
            return "Unknown SSL error code";
    }
}

HandshakeResult handshake_consumed_bytes_and_needs_more_peer_data(size_t consumed) noexcept {
    return {consumed, 0, HandshakeResult::State::NeedsMorePeerData};
}

HandshakeResult handshake_consumed_bytes_and_is_complete(size_t consumed) noexcept {
    return {consumed, 0, HandshakeResult::State::Done};
}

HandshakeResult handshaked_bytes(size_t consumed, size_t produced, HandshakeResult::State state) noexcept {
    return {consumed, produced, state};
}

HandshakeResult handshake_completed() noexcept {
    return {0, 0, HandshakeResult::State::Done};
}

HandshakeResult handshake_failed() noexcept {
    return {0, 0, HandshakeResult::State::Failed};
}

EncodeResult encode_failed() noexcept {
    return {0, 0, true};
}

EncodeResult encoded_bytes(size_t consumed, size_t produced) noexcept {
    return {consumed, produced, false};
}

DecodeResult decode_failed() noexcept {
    return {0, 0, DecodeResult::State::Failed};
}

DecodeResult decoded_frames_with_plaintext_bytes(size_t produced_bytes) noexcept {
    return {0, produced_bytes, DecodeResult::State::OK};
}

DecodeResult decode_needs_more_peer_data() noexcept {
    return {0, 0, DecodeResult::State::NeedsMorePeerData};
}

DecodeResult decoded_bytes(size_t consumed, size_t produced, DecodeResult::State state) noexcept {
    return {consumed, produced, state};
}

BioPtr new_tls_frame_mutable_memory_bio() {
    BioPtr bio(new_mutable_direct_buffer_bio());
    if (!bio) {
        throw CryptoException("new_mutable_direct_buffer_bio() failed; out of memory?");
    }
    return bio;
}

BioPtr new_tls_frame_const_memory_bio() {
    BioPtr bio(new_const_direct_buffer_bio());
    if (!bio) {
        throw CryptoException("new_const_direct_buffer_bio() failed; out of memory?");
    }
    return bio;
}

vespalib::string ssl_error_from_stack() {
    char buf[256];
    ERR_error_string_n(ERR_get_error(), buf, sizeof(buf));
    return vespalib::string(buf);
}

} // anon ns

OpenSslCryptoCodecImpl::OpenSslCryptoCodecImpl(::SSL_CTX& ctx, Mode mode)
    : _ssl(::SSL_new(&ctx)),
      _mode(mode)
{
    if (!_ssl) {
        throw CryptoException("Failed to create new SSL from SSL_CTX");
    }
    /*
     * We use two separate memory BIOs rather than a BIO pair for writing and
     * reading ciphertext, respectively. This is because it _seems_ quite
     * a bit more straight forward to implement a full duplex API with two
     * separate BIOs, but there is little available documentation as to the
     * 'hows' and 'whys' around this.
     *
     * Our BIOs are used as follows:
     *
     * Handshakes may use both BIOs opaquely:
     *
     *  handshake() : SSL_do_handshake()  --(_output_bio ciphertext)--> BIO_read  --> [peer]
     *              : SSL_do_handshake() <--(_input_bio ciphertext)--   BIO_write <-- [peer]
     *
     * Once handshaking is complete, the input BIO is only used for decodes and the output
     * BIO is only used for encodes. We explicitly disallow TLS renegotiation, both for
     * the sake of simplicity and for added security (renegotiation is a bit of a rat's nest).
     *
     *  encode() : SSL_write(plaintext) --(_output_bio ciphertext)--> BIO_read  --> [peer]
     *  decode() : SSL_read(plaintext) <--(_input_bio ciphertext)--   BIO_write <-- [peer]
     *
     */
    BioPtr tmp_input_bio  = new_tls_frame_const_memory_bio();
    BioPtr tmp_output_bio = new_tls_frame_mutable_memory_bio();
    // Connect BIOs used internally by OpenSSL. This transfers ownership. No return values to check.
#if (OPENSSL_VERSION_NUMBER >= 0x10100000L)
    ::SSL_set0_rbio(_ssl.get(), tmp_input_bio.get());
    ::SSL_set0_wbio(_ssl.get(), tmp_output_bio.get());
#else
    ::SSL_set_bio(_ssl.get(), tmp_input_bio.get(), tmp_output_bio.get());
#endif
    _input_bio  = tmp_input_bio.release();
    _output_bio = tmp_output_bio.release();
    if (_mode == Mode::Client) {
        ::SSL_set_connect_state(_ssl.get());
    } else {
        ::SSL_set_accept_state(_ssl.get());
    }
}

// TODO remove spammy logging once code is stable

HandshakeResult OpenSslCryptoCodecImpl::handshake(const char* from_peer, size_t from_peer_buf_size,
                                                  char* to_peer, size_t to_peer_buf_size) noexcept {
    LOG_ASSERT(verify_buf(from_peer, from_peer_buf_size) && verify_buf(to_peer, to_peer_buf_size));

    if (SSL_is_init_finished(_ssl.get())) {
        return handshake_completed();
    }
    ConstBufferViewGuard const_view_guard(*_input_bio, from_peer, from_peer_buf_size);
    MutableBufferViewGuard mut_view_guard(*_output_bio, to_peer, to_peer_buf_size);

    const auto consume_res = do_handshake_and_consume_peer_input_bytes();
    LOG_ASSERT(consume_res.bytes_produced == 0);
    if (consume_res.failed()) {
        return consume_res;
    }
    // SSL_do_handshake() might have produced more data to send. Note: handshake may
    // be complete at this point.
    int produced = BIO_pending(_output_bio);
    return handshaked_bytes(consume_res.bytes_consumed, static_cast<size_t>(produced), consume_res.state);
}

HandshakeResult OpenSslCryptoCodecImpl::do_handshake_and_consume_peer_input_bytes() noexcept {
    // Assumption: SSL_do_handshake will place all required outgoing handshake
    // data in the output memory BIO without requiring WANT_WRITE.
    // TODO test multi-frame sized handshake
    const long pending_read_before = BIO_pending(_input_bio);

    ::ERR_clear_error();
    int ssl_result = ::SSL_do_handshake(_ssl.get());
    ssl_result = ::SSL_get_error(_ssl.get(), ssl_result);

    const long consumed = pending_read_before - BIO_pending(_input_bio);
    LOG_ASSERT(consumed >= 0);

    if (ssl_result == SSL_ERROR_WANT_READ) {
        LOG(spam, "SSL_do_handshake() returned SSL_ERROR_WANT_READ");

        return handshake_consumed_bytes_and_needs_more_peer_data(static_cast<size_t>(consumed));
    } else if (ssl_result == SSL_ERROR_NONE) {
        // At this point SSL_do_handshake has stated it does not need any more peer data, i.e.
        // the handshake is complete.
        if (!SSL_is_init_finished(_ssl.get())) {
            LOG(error, "SSL handshake is not completed even though no more peer data is requested");
            return handshake_failed();
        }
        LOG(debug, "SSL_do_handshake() is complete, using protocol %s", SSL_get_version(_ssl.get()));
        return handshake_consumed_bytes_and_is_complete(static_cast<size_t>(consumed));
    } else {
        LOG(error, "SSL_do_handshake() returned unexpected error: %s (%s)",
            ssl_error_to_str(ssl_result), ssl_error_from_stack().c_str());
        return handshake_failed();
    }
}

EncodeResult OpenSslCryptoCodecImpl::encode(const char* plaintext, size_t plaintext_size,
                                            char* ciphertext, size_t ciphertext_size) noexcept {
    LOG_ASSERT(verify_buf(plaintext, plaintext_size) && verify_buf(ciphertext, ciphertext_size));

    if (!SSL_is_init_finished(_ssl.get())) {
        LOG(error, "OpenSslCryptoCodecImpl::encode() called before handshake completed");
        return encode_failed();
    }

    MutableBufferViewGuard mut_view_guard(*_output_bio, ciphertext, ciphertext_size);
    // _input_bio not read from here.

    size_t bytes_consumed = 0;
    if (plaintext_size != 0) {
        ::ERR_clear_error();
        int to_consume = static_cast<int>(std::min(plaintext_size, MaximumFramePlaintextSize));
        // SSL_write encodes plaintext to ciphertext and writes to _output_bio
        int consumed = ::SSL_write(_ssl.get(), plaintext, to_consume);
        LOG(spam, "After SSL_write() -> %d _output_bio pending=%d", consumed, BIO_pending(_output_bio));
        if (consumed < 0) {
            int ssl_error = ::SSL_get_error(_ssl.get(), consumed);
            LOG(error, "SSL_write() failed to write frame, got error %s (%s)",
                ssl_error_to_str(ssl_error), ssl_error_from_stack().c_str());
            // TODO explicitly detect and log TLS renegotiation error (SSL_ERROR_WANT_READ)?
            return encode_failed();
        } else if (consumed != to_consume) {
            LOG(error, "SSL_write() returned OK but did not consume all requested plaintext");
            return encode_failed();
        }
        bytes_consumed = static_cast<size_t>(consumed);
    }
    int produced = BIO_pending(_output_bio);
    return encoded_bytes(bytes_consumed, static_cast<size_t>(produced));
}
DecodeResult OpenSslCryptoCodecImpl::decode(const char* ciphertext, size_t ciphertext_size,
                                            char* plaintext, size_t plaintext_size) noexcept {
    LOG_ASSERT(verify_buf(ciphertext, ciphertext_size) && verify_buf(plaintext, plaintext_size));

    if (!SSL_is_init_finished(_ssl.get())) {
        LOG(error, "OpenSslCryptoCodecImpl::decode() called before handshake completed");
        return decode_failed();
    }
    ConstBufferViewGuard const_view_guard(*_input_bio, ciphertext, ciphertext_size);
    // _output_bio not written to here

    const int input_pending_before = BIO_pending(_input_bio);
    auto produce_res = drain_and_produce_plaintext_from_ssl(plaintext, static_cast<int>(plaintext_size));
    const int input_pending_after = BIO_pending(_input_bio);

    LOG_ASSERT(input_pending_before >= input_pending_after);
    const int consumed = input_pending_before - input_pending_after;
    LOG(spam, "decode: consumed %d bytes (ciphertext buffer %d -> %d bytes), produced %zu bytes. Need read: %s",
        consumed, input_pending_before, input_pending_after, produce_res.bytes_produced,
        (produce_res.state == DecodeResult::State::NeedsMorePeerData) ? "yes" : "no");
    return decoded_bytes(static_cast<size_t>(consumed), produce_res.bytes_produced, produce_res.state);
}

DecodeResult OpenSslCryptoCodecImpl::drain_and_produce_plaintext_from_ssl(
        char* plaintext, size_t plaintext_size) noexcept {
    ::ERR_clear_error();
    // SSL_read() is named a bit confusingly. We read _from_ the SSL-internal state
    // via the input BIO _into_ to the receiving plaintext buffer.
    // This may consume the entire, parts of, or none of the input BIO's data,
    // depending on how much TLS frame data is available and its size relative
    // to the receiving plaintext buffer.
    int produced = ::SSL_read(_ssl.get(), plaintext, static_cast<int>(plaintext_size));
    if (produced > 0) {
        // At least 1 frame decoded successfully.
        return decoded_frames_with_plaintext_bytes(static_cast<size_t>(produced));
    } else {
        int ssl_error = ::SSL_get_error(_ssl.get(), produced);
        switch (ssl_error) {
        case SSL_ERROR_WANT_READ:
            // SSL_read() was not able to decode a full frame with the ciphertext that
            // we've fed it thus far; caller must feed it some and then try again.
            LOG(spam, "SSL_read() returned SSL_ERROR_WANT_READ, must get more ciphertext");
            return decode_needs_more_peer_data();
        default:
            LOG(error, "SSL_read() returned unexpected error: %s (%s)",
                ssl_error_to_str(ssl_error), ssl_error_from_stack().c_str());
            return decode_failed();
        }
    }
}

}

// External references:
//  [0] http://openssl.6102.n7.nabble.com/nonblocking-implementation-question-tp1728p1732.html
//  [1] https://github.com/grpc/grpc/blob/master/src/core/tsi/ssl_transport_security.cc
