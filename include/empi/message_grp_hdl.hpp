#ifndef EMPI_PROJECT_MGH_HPP
#define EMPI_PROJECT_MGH_HPP

#include "mpi.h"
#include <memory>
#include <limits>

#include <empi/request_pool.hpp>
#include <empi/type_traits.hpp>
#include <empi/tag.hpp>

#include <empi/async_event.hpp>
#include <empi/utils.hpp>
#include <empi/datatype.hpp>
#include <empi/defines.hpp>
#include <empi/layouts.hpp>
#include <empi/compact.hpp>
#include <utility>

namespace empi{


	template<typename T1, Tag TAG = NOTAG, std::size_t SIZE = 0>
	class MessageGroupHandler{

	  	using T = details::remove_all_t<T1>;

		public:
		  explicit MessageGroupHandler(MPI_Comm comm, std::shared_ptr<request_pool> _request_pool) : communicator(comm), _request_pool(std::move(_request_pool)) {
			// MPI_Datatype type = details::mpi_type<T>::get_type();
			max_tag = std::numeric_limits<int>::max(); //TODO: Remove this
			// MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &max_tag, &flag);
			// if(!flag){
			//   max_tag = -1;
			// }
			// EMPI_CHECKTYPE(type); //TODO: exceptions?
		  }

		  MessageGroupHandler(const MessageGroupHandler& chg) = default;
		  MessageGroupHandler(MessageGroupHandler&& chg)  noexcept = default;
		
		 // -------------- UTILITY -----------------------------

		int inline barrier() const {
			return MPI_Barrier(communicator);
		}

		void waitall() {
			_request_pool->waitall();
		}

		  // -------------- SEND -----------------------------------------
			
		  template<typename K>
		  int inline _send_impl(K&& data, const int dest, const size_t size, const Tag tag) const {
				using element_type = details::get_true_type_t<K>;
				if constexpr (!details::is_mdspan<K>){
					return EMPI_SEND(empi_byte_cast(details::get_underlying_pointer(std::forward<K>(data))), size * details::size_of<element_type>,  MPI_BYTE, dest, tag.value, communicator);
				}
				else{
					auto&& ptr = layouts::compact(std::forward<K>(data));
					return EMPI_SEND(empi_byte_cast(ptr.get()), size * details::size_of<element_type>,  MPI_BYTE, dest, tag.value, communicator);
				}	
		  }

		  template<typename K>
		  requires (SIZE == NOSIZE) && (TAG == NOTAG)
		  int send(K&& data, int dest, const size_t size, Tag tag) const {
			details::checktag<details::mpi_function::send>(tag.value, max_tag);
			//TODO: Check type...
			//TODO: Check size...
			return _send_impl(std::forward<K>(data), dest, size, tag);
		  }

		  template<typename K>
		  requires (SIZE > 0) && (TAG != -1)
		  int send(K&& data, int dest){
		  	return _send_impl(data, dest, SIZE, TAG);
		  }

		  template<typename K>
		  requires (SIZE > 0) && (TAG == NOTAG)
		  int send(K&& data, int dest, Tag tag){
			details::checktag<details::mpi_function::send>(tag.value, max_tag);
			return _send_impl(data, dest, SIZE, tag);
		  }

		  template<typename K>
		  requires (SIZE == NOSIZE) && (TAG != -1)
		  int inline send(K&& data, int dest, const size_t size){
			// Check size...
			return _send_impl(data, dest, size, TAG);
		  }

		  // ---------------------------- END SEND --------------------------------

		  // ---------------------------- START RECV ------------------------------
		  
		  template<typename K>
		  inline int _recv_impl(K&& data, const int src, const size_t size, const Tag tag, MPI_Status& status){
			using element_type = details::get_true_type_t<K>;
			return EMPI_RECV(empi_byte_cast(details::get_underlying_pointer(data)), size * details::size_of<element_type>,  MPI_BYTE, src, tag.value, communicator, &status);
		  }

		  template<typename K>
		  requires (SIZE == NOSIZE) && (TAG == NOTAG)
		  int recv(K&& data, const int src, const size_t size, Tag tag, MPI_Status& status){
			details::checktag<details::mpi_function::recv>(tag.value, max_tag);
			//Check size...
			return _recv_impl(std::forward<K>(data), src, size, tag, status);
		  }


		  template<typename K>
		  requires (SIZE > 0) && (TAG.value >= -1)
		  int recv(K&& data, const int src, MPI_Status& status){
			return _recv_impl(data, src, SIZE, TAG, status);
		  }

		  template<typename K>
		  requires (SIZE == NOSIZE) && (TAG.value >= -1)
		  int inline recv(K&& data, const int src, const size_t size, MPI_Status& status){
			// Check size...
			return _recv_impl(data, src, size, TAG, status);
		  }


		  template<typename K>
		  requires (SIZE > 0) && (TAG == NOTAG)
		  int recv(K&& data, const int src, Tag tag, MPI_Status& status){
			details::checktag<details::mpi_function::recv>(tag.value, max_tag);
			return _recv_impl(data, src, SIZE, tag, status);
		  }

		  // ------------------------- END RECV -----------------------------


		  // ------------------------- START ISEND --------------------------
		  template<typename K>
		  requires (SIZE > 0) && (TAG != -1)
		  inline std::shared_ptr<async_event>& _isend_impl(K&& data, const int dest, const size_t size, const Tag tag){
			auto&& event = _request_pool->get_req();
			using element_type = details::get_true_type_t<K>;
			if constexpr (!details::is_mdspan<K>){
				return EMPI_ISEND(empi_byte_cast(details::get_underlying_pointer(std::forward<K>(data))), size * details::size_of<element_type>,  MPI_BYTE, dest, tag.value, communicator, event->request.get());
			}
			else{
				auto&& ptr = layouts::compact(std::forward<K>(data));
				return EMPI_ISEND(empi_byte_cast(ptr.get()), size * details::size_of<element_type>,  MPI_BYTE, dest, tag.value, communicator);
			}
			return event;
		  }

		  template<typename K>
		  requires (SIZE > 0) && (TAG != -1)
		  std::shared_ptr<async_event>& Isend(K&& data, int dest){
			// Check type
			return _isend_impl(std::forward<K>(data), dest, SIZE, TAG);
		  }


		  template<typename K>
		  requires (SIZE == NOSIZE) && (TAG != -1)
		  std::shared_ptr<async_event>& Isend(K&& data, int dest, const size_t size){
			// Check size...
			return _isend_impl(std::forward<K>(data), dest, size, TAG);
		  }

		  template<typename K>
		  requires (SIZE > 0) && (TAG == NOTAG)
		  std::shared_ptr<async_event>& Isend(K&& data, int dest, Tag tag){
			details::checktag<details::mpi_function::isend>(tag.value, max_tag);
			return _isend_impl(std::forward<K>(data), dest, SIZE, tag);
		  }

		  template<typename K>
		  requires (SIZE == NOSIZE) && (TAG == NOTAG)
		  std::shared_ptr<async_event>& Isend(K&& data, int dest, const size_t size, Tag tag){
			//Check size...
			details::checktag<details::mpi_function::isend>(tag.value, max_tag);
			return _isend_impl(std::forward<K>(data), dest, size, tag);
		  }

	  // ------------------------- END ISEND -----------------------------


	  // ------------------------- START URECV --------------------------
		template<typename K>
		requires (SIZE > 0) && (TAG >= -2)
		inline std::shared_ptr<async_event>&  _irecv_impl(K&& data, const int src, const size_t size, const Tag tag) const {
		  	auto&& event = _request_pool->get_req();
			using element_type = details::get_true_type_t<K>;
			return EMPI_IRECV(empi_byte_cast(details::get_underlying_pointer(data)), size * details::size_of<element_type>,  MPI_BYTE, src, tag.value, communicator, event->request.get());
		}

		template<typename K>
		requires (SIZE > 0) && (TAG >= -2)
		std::shared_ptr<async_event>& Irecv(K&& data, const int src) const {
		 return _irecv_impl(data, src, SIZE, TAG);
		}

		template<typename K>
		requires (SIZE == NOSIZE) && (TAG >= -2)
		std::shared_ptr<async_event>& Irecv(K&& data, const int src, const size_t size) const {
		 return _irecv_impl(data, src, size, TAG);
		}

		template<typename K>
		requires (SIZE > 0) && (TAG == NOTAG)
		std::shared_ptr<async_event>& Irecv(K&& data,const int src, const Tag tag) const {
		  details::checktag<details::mpi_function::irecv>(tag.value, max_tag);
		  return _irecv_impl(data, src, SIZE, tag);
		}

		template<typename K>
		requires (SIZE == NOSIZE) && (TAG == NOTAG)
		std::shared_ptr<async_event>& Irecv(K&& data, const int src,const size_t size,const Tag tag) const {
		  details::checktag<details::mpi_function::irecv>(tag.value, max_tag);
		  return _irecv_impl(data, src, size, tag);
		}

	  // ------------------------- END URECV --------------------------
	  // ------------------------- BCAST --------------------------

   	  template<typename K>
	  int _bcast_impl(K&& data, const int root, const size_t size){
		using element_type = details::get_true_type_t<K>;
		element_type* ptr = details::get_underlying_pointer(std::forward<K>(data));
		return EMPI_BCAST(empi_byte_cast(ptr), size * details::size_of<element_type>, MPI_BYTE,root,communicator);		
	  }

      template<typename K>
	  requires (SIZE == NOSIZE)
	  int Bcast(K&& data, const int root, const size_t size){
		// Check size...
		return _bcast_impl(std::forward<K>(data), root, size);
	  }

	  template<typename K>
	  requires (SIZE > 0)
	  int Bcast(K&& data, const int root){
	  	return _bcast_impl(std::forward<K>(data), root, SIZE);
	  }

	  // ------------------------- END BCAST --------------------------
	  // ------------------------- IBCAST --------------------------

	  template<typename K>
	  requires (details::is_valid_container<T,K> || details::is_valid_pointer<T,K>) && (SIZE > 0)
	  std::shared_ptr<async_event>& Ibcast(K&& data, const int root){
		auto&& event = _request_pool->get_req();
		event->res = EMPI_IBCAST(details::get_underlying_pointer(data), SIZE, details::mpi_type<T>::get_type(),root,communicator, event->get_request());
		return event;
	  }

	  template<typename K>
	  requires (details::is_valid_container<T,K> || details::is_valid_pointer<T,K>) && (SIZE == NOSIZE)
	  std::shared_ptr<async_event>& Ibcast(K&& data, const int root, const size_t size){
		auto&& event = _request_pool->get_req();
		event->res = EMPI_IBCAST(details::get_underlying_pointer(data), size, details::mpi_type<T>::get_type(),root,communicator, event->get_request());
		return event;
	  }

	  // ------------------------- END IBCAST --------------------------
	  // ------------------------- ALLREDUCE --------------------------

	  template<typename K>
	  requires (details::is_valid_container<T,K> || details::is_valid_pointer<T,K>) && (SIZE > 0)
	  int Allreduce(K&& sendbuf, K&& recvbuf, MPI_Op op){
		return EMPI_ALLREDUCE(details::get_underlying_pointer(sendbuf),details::get_underlying_pointer(recvbuf),SIZE,details::mpi_type<T>::get_type(),op,communicator);
	  }

	  template<typename K>
	  requires (details::is_valid_container<T,K> || details::is_valid_pointer<T,K>) && (SIZE == NOSIZE)
	  int Allreduce(K&& sendbuf, K&& recvbuf, const size_t size, MPI_Op op){
			return EMPI_ALLREDUCE(sendbuf,recvbuf,size,details::mpi_type<T>::get_type(),op,communicator);
	  }

	  // ------------------------- END ALLREDUCE --------------------------
	  // ------------------------- GATHERV --------------------------
	template<typename K>
	  requires (details::is_valid_container<T,K> || details::is_valid_pointer<T,K>)
	  int gatherv(const int root, K&& sendbuf,int sendcount, K&& recvbuf, int* recvcounts, int* displacements){
		return EMPI_GATHERV(details::get_underlying_pointer(sendbuf), 
						   sendcount,
						   details::mpi_type<T>::get_type(),
						   details::get_underlying_pointer(recvbuf),
						   recvcounts,
						   displacements,
						   details::mpi_type<T>::get_type(),
						   root,
						   communicator);
	  }
	  // ------------------------- END ALLREDUCE --------------------------


		private:
			MPI_Comm communicator;
			std::shared_ptr<request_pool> _request_pool;
			int max_tag;
	};

}
#endif // __MESSAGE_GRP_HDL_H__
