/*
 * Copyright (c) 2022-2023 University of Salerno, Italy. All rights reserved.
 */

#ifndef EMPI_PROJECT_MGH_HPP
#define EMPI_PROJECT_MGH_HPP

#include "mpi.h"
#include <limits>
#include <memory>

#include <empi/request_pool.hpp>
#include <empi/tag.hpp>
#include <empi/type_traits.hpp>

#include <empi/async_event.hpp>
#include <empi/compact/compact.hpp>
#include <empi/datatype.hpp>
#include <empi/defines.hpp>
#include <empi/layouts.hpp>
#include <empi/utils.hpp>
#include <utility>

namespace empi {


template<typename T1, Tag TAG = NOTAG, std::size_t SIZE = 0>
class MessageGroupHandler {
    using T = details::remove_all_t<T1>;

  public:
    explicit MessageGroupHandler(MPI_Comm comm, std::shared_ptr<request_pool> _request_pool, int rank, int size)
        : m_communicator(comm), m_request_pool(std::move(_request_pool)), m_rank(rank), m_size(size) {
        // MPI_Datatype type = details::mpi_type<T>::get_type();
        m_max_tag = std::numeric_limits<int>::max(); // TODO: Remove this
        // MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &max_tag, &flag);
        // if(!flag){
        //   max_tag = -1;
        // }
        // EMPI_CHECKTYPE(type); //TODO: exceptions?
    }

    MessageGroupHandler(const MessageGroupHandler &chg) = default;
    MessageGroupHandler(MessageGroupHandler &&chg) noexcept = default;

    // -------------- UTILITY -----------------------------

    int inline barrier() const { return MPI_Barrier(m_communicator); }

    void waitall() { m_request_pool->waitall(); }

    // -------------- SEND -----------------------------------------

    template<typename K>
    int inline _send_impl(K &&data, const int dest, const size_t size, const Tag tag) const {
        using element_type = details::get_true_type_t<K>;
        auto &&ptr = details::get_underlying_pointer(std::forward<K>(data), true);
        return EMPI_SEND(
            empi_byte_cast(ptr), size * details::size_of<element_type>, MPI_BYTE, dest, tag.value, m_communicator);
    }

    template<typename K>
        requires(SIZE == NOSIZE) && (TAG == NOTAG)
    int send(K &&data, int dest, const size_t size, Tag tag) const {
        details::checktag<details::mpi_function::send>(tag.value, m_max_tag);
        // TODO: Check type...
        // TODO: Check size...
        return _send_impl(std::forward<K>(data), dest, size, tag);
    }

    template<typename K>
        requires(SIZE > 0) && (TAG != -1)
    int send(K &&data, int dest) {
        return _send_impl(data, dest, SIZE, TAG);
    }

    template<typename K>
        requires(SIZE > 0) && (TAG == NOTAG)
    int send(K &&data, int dest, Tag tag) {
        details::checktag<details::mpi_function::send>(tag.value, m_max_tag);
        return _send_impl(data, dest, SIZE, tag);
    }

    template<typename K>
        requires(SIZE == NOSIZE) && (TAG != -1)
    int inline send(K &&data, int dest, const size_t size) {
        // Check size...
        return _send_impl(data, dest, size, TAG);
    }

    // ---------------------------- END SEND --------------------------------

    // ---------------------------- START RECV ------------------------------

    template<typename K>
    inline int _recv_impl(K &&data, const int src, const size_t size, const Tag tag, MPI_Status &status) {
        using element_type = details::get_true_type_t<K>;
        return EMPI_RECV(empi_byte_cast(details::get_underlying_pointer(data)), size * details::size_of<element_type>,
            MPI_BYTE, src, tag.value, m_communicator, &status);
    }

    template<typename K>
        requires(SIZE == NOSIZE) && (TAG == NOTAG)
    int recv(K &&data, const int src, const size_t size, Tag tag, MPI_Status &status) {
        details::checktag<details::mpi_function::recv>(tag.value, m_max_tag);
        // Check size...
        return _recv_impl(std::forward<K>(data), src, size, tag, status);
    }


    template<typename K>
        requires(SIZE > 0) && (TAG.value >= -1)
    int recv(K &&data, const int src, MPI_Status &status) {
        return _recv_impl(data, src, SIZE, TAG, status);
    }

    template<typename K>
        requires(SIZE == NOSIZE) && (TAG.value >= -1)
    int inline recv(K &&data, const int src, const size_t size, MPI_Status &status) {
        // Check size...
        return _recv_impl(data, src, size, TAG, status);
    }


    template<typename K>
        requires(SIZE > 0) && (TAG == NOTAG)
    int recv(K &&data, const int src, Tag tag, MPI_Status &status) {
        details::checktag<details::mpi_function::recv>(tag.value, m_max_tag);
        return _recv_impl(data, src, SIZE, tag, status);
    }

    // ------------------------- END RECV -----------------------------


    // ------------------------- START ISEND --------------------------
    template<typename K>
    inline std::shared_ptr<async_event> &_isend_impl(K &&data, const int dest, const size_t size, const Tag tag) {
        auto &&event = m_request_pool->get_req();
        using element_type = details::get_true_type_t<K>;
        auto &&ptr = details::get_underlying_pointer(std::forward<K>(data));
        EMPI_ISEND(empi_byte_cast(ptr), size * details::size_of<element_type>, MPI_BYTE, dest, tag.value, m_communicator,
            event->request.get());
        return event;
    }

    template<typename K>
        requires(SIZE > 0) && (TAG != -1)
    std::shared_ptr<async_event> &Isend(K &&data, int dest) {
        // Check type
        return _isend_impl(std::forward<K>(data), dest, SIZE, TAG);
    }


    template<typename K>
        requires(SIZE == NOSIZE) && (TAG != -1)
    std::shared_ptr<async_event> &Isend(K &&data, int dest, const size_t size) {
        // Check size...
        return _isend_impl(std::forward<K>(data), dest, size, TAG);
    }

    template<typename K>
        requires(SIZE > 0) && (TAG == NOTAG)
    std::shared_ptr<async_event> &Isend(K &&data, int dest, Tag tag) {
        details::checktag<details::mpi_function::isend>(tag.value, m_max_tag);
        return _isend_impl(std::forward<K>(data), dest, SIZE, tag);
    }

    template<typename K>
        requires(SIZE == NOSIZE) && (TAG == NOTAG)
    std::shared_ptr<async_event> &Isend(K &&data, int dest, const size_t size, Tag tag) {
        // Check size...
        details::checktag<details::mpi_function::isend>(tag.value, m_max_tag);
        return _isend_impl(std::forward<K>(data), dest, size, tag);
    }

    // ------------------------- END ISEND -----------------------------


    // ------------------------- START URECV --------------------------
    template<typename K>
    inline std::shared_ptr<async_event> &_irecv_impl(K &&data, const int src, const size_t size, const Tag tag) const {
        auto &&event = m_request_pool->get_req();
        using element_type = details::get_true_type_t<K>;
        EMPI_IRECV(empi_byte_cast(details::get_underlying_pointer(data)), size * details::size_of<element_type>,
            MPI_BYTE, src, tag.value, m_communicator, event->request.get());
        return event;
    }

    template<typename K>
        requires(SIZE > 0) && (TAG >= -2)
    std::shared_ptr<async_event> &Irecv(K &&data, const int src) const {
        return _irecv_impl(data, src, SIZE, TAG);
    }

    template<typename K>
        requires(SIZE == NOSIZE) && (TAG >= -2)
    std::shared_ptr<async_event> &Irecv(K &&data, const int src, const size_t size) const {
        return _irecv_impl(data, src, size, TAG);
    }

    template<typename K>
        requires(SIZE > 0) && (TAG == NOTAG)
    std::shared_ptr<async_event> &Irecv(K &&data, const int src, const Tag tag) const {
        details::checktag<details::mpi_function::irecv>(tag.value, m_max_tag);
        return _irecv_impl(data, src, SIZE, tag);
    }

    template<typename K>
        requires(SIZE == NOSIZE) && (TAG == NOTAG)
    std::shared_ptr<async_event> &Irecv(K &&data, const int src, const size_t size, const Tag tag) const {
        details::checktag<details::mpi_function::irecv>(tag.value, m_max_tag);
        return _irecv_impl(data, src, size, tag);
    }

    // ------------------------- END URECV --------------------------
    // ------------------------- BCAST --------------------------

    template<typename K>
    int _bcast_impl(K &&data, const int root, const size_t size) {
        using element_type = details::get_true_type_t<K>;
        auto &&ptr = details::get_underlying_pointer(std::forward<K>(data), root == m_rank);
        return EMPI_BCAST(empi_byte_cast(ptr), size * details::size_of<element_type>, MPI_BYTE, root, m_communicator);
    }

    template<typename K>
        requires(SIZE == NOSIZE)
    int Bcast(K &&data, const int root, const size_t size) {
        // Check size...
        return _bcast_impl(std::forward<K>(data), root, size);
    }

    template<typename K>
        requires(SIZE > 0)
    int Bcast(K &&data, const int root) {
        return _bcast_impl(std::forward<K>(data), root, SIZE);
    }

    // ------------------------- END BCAST --------------------------
    // ------------------------- IBCAST --------------------------
    template<typename K>
    std::shared_ptr<async_event> &_ibcast_impl(K &&data, const int root, const size_t size) {
        using element_type = details::get_true_type_t<K>;
        auto &&event = m_request_pool->get_req();
        auto &&ptr = details::get_underlying_pointer(std::forward<K>(data), root == m_rank);
        EMPI_IBCAST(empi_byte_cast(ptr), size * details::size_of<element_type>, MPI_BYTE, root, m_communicator,
            event->get_request());
        return event;
    }

    template<typename K>
        requires(SIZE == NOSIZE)
    auto &Ibcast(K &&data, const int root, const size_t size) {
        // Check size...
        return _ibcast_impl(std::forward<K>(data), root, size);
    }

    template<typename K>
        requires(SIZE > 0)
    auto &Ibcast(K &&data, const int root) {
        return _ibcast_impl(std::forward<K>(data), root, SIZE);
    }

    // ------------------------- END IBCAST --------------------------
    // ------------------------- ALLREDUCE --------------------------

    /**
      Reductions introduce a lot of problems.
      MPI_BYTE cannot be used in reductions for several reasons:
          1) Reduction algorithms split data across nodes and apply operators on them, which is impossible
             with bytes as you have to cast-back the data to the actual elements to manipulate it, and
             MPI does not hold that information
          2) If splitting size is not multiple of the orginal datatype, the last element of each phase will
             be truncated
          3) MPI used types to allocate identity values, the minimum and the maximum value and so on.
      So, for now we will just skip reductions and stick with the old techniques (MPI base types and no layouts)
    */
    template<typename K>
        requires(details::is_valid_container<T, K> || details::is_valid_pointer<T, K>) && (SIZE > 0)
    int Allreduce(K &&sendbuf, K &&recvbuf, MPI_Op op) {
        return EMPI_ALLREDUCE(details::get_underlying_pointer(sendbuf), details::get_underlying_pointer(recvbuf), SIZE,
            details::mpi_type<T>::get_type(), op, m_communicator);
    }

    template<typename K>
        requires(details::is_valid_container<T, K> || details::is_valid_pointer<T, K>) && (SIZE == NOSIZE)
    int Allreduce(K &&sendbuf, K &&recvbuf, const size_t size, MPI_Op op) {
        return EMPI_ALLREDUCE(sendbuf, recvbuf, size, details::mpi_type<T>::get_type(), op, m_communicator);
    }

    // ------------------------- END ALLREDUCE --------------------------

    // ------------------------- GATHERV --------------------------
    template<typename K>
    int _gatherv_impl(const int root, K &&sendbuf, int sendcount, K &&recvbuf,
        details::typed_range<int> auto &&recvcounts, details::typed_range<int> auto &&displacements) {
        using send_element_type = details::get_true_type_t<K>;
        using recv_element_type = details::get_true_type_t<K>;
        auto &&send_ptr = details::get_underlying_pointer(std::forward<K>(sendbuf), root == m_rank);
        auto &&recv_ptr = details::get_underlying_pointer(std::forward<K>(recvbuf));

        // Each node receives recvcount * size bytes
        constexpr auto byte_conv = [](auto &e) { const_cast<int &>(e) = e * details::size_of<send_element_type>; };
        std::ranges::for_each(recvcounts, byte_conv);
        std::ranges::for_each(displacements, byte_conv);

        return EMPI_GATHERV(empi_byte_cast(send_ptr), sendcount * details::size_of<send_element_type>, MPI_BYTE,
            empi_byte_cast(recv_ptr), std::ranges::data(recvcounts), std::ranges::data(displacements), MPI_BYTE, root,
            m_communicator);
    }


    template<typename K, details::typed_range<int> J>
    int gatherv(const int root, K &&sendbuf, int sendcount, K &&recvbuf, J &&recvcounts, J &&displacements) {
        // Check sizes...
        return _gatherv_impl(root, std::forward<K>(sendbuf), sendcount, std::forward<K>(recvbuf),
            std::forward<J>(recvcounts), std::forward<J>(displacements));
    }

    template<typename K>
    int gatherv(const int root, K &&sendbuf, int sendcount, K &&recvbuf, auto &&recvcounts, auto &&displacements) {
        // Check sizes...
        return _gatherv_impl(root, std::forward<K>(sendbuf), sendcount, std::forward<K>(recvbuf),
            std::span(details::get_underlying_pointer(std::forward<decltype(recvcounts)>(recvcounts)).get(), m_size),
            std::span(
                details::get_underlying_pointer(std::forward<decltype(displacements)>(displacements)).get(), m_size));
    }
    // ------------------------- END GATHERV --------------------------

    template<typename T, typename K>
    int _allgather_impl(T &&sendbuf, int sendcount, K &&recvbuf, int recvcounts) {
        using send_element_type = details::get_true_type_t<K>;
        using recv_element_type = details::get_true_type_t<K>;
        auto send_ptr = details::get_underlying_pointer(std::forward<T>(sendbuf), true);
        auto recv_ptr = details::get_underlying_pointer(std::forward<K>(recvbuf));

        return EMPI_ALLGATHER(empi_byte_cast(send_ptr), sendcount * details::size_of<send_element_type>, MPI_BYTE,
            empi_byte_cast(recv_ptr), recvcounts * details::size_of<recv_element_type>, MPI_BYTE, m_communicator);
    }

    template<typename T, typename K>
    int Allgather(T &&sendbuf, int sendcount, K &&recvbuf, int recvcounts) {
        return _allgather_impl(std::forward<T>(sendbuf), sendcount, std::forward<K>(recvbuf), recvcounts);
    }


  private:
    MPI_Comm m_communicator;
    std::shared_ptr<request_pool> m_request_pool;
    int m_max_tag;
    const int m_rank;
    const int m_size;
};

} // namespace empi
#endif // __MESSAGE_GRP_HDL_H__
