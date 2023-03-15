#include "utils.hpp"
#include <cstdint>
#include <experimental/mdspan>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <empi/empi.hpp>
#include <empi/utils.hpp>

namespace stdex = std::experimental;

// TEST_CASE("Send and receive a view with no compile-time parameters", "[mgh][layouts]"){
// 	empi::Context ctx{nullptr,nullptr};
// 	auto mg = ctx.create_message_group(MPI_COMM_WORLD);

// 	std::vector<int> v(10);
// 	if(mg->rank() == 0)
// 		std::iota(v.begin(), v.end(), 0);

// 	mg->run([&](empi::MessageGroupHandler<float>& mgh){
// 		if(mg->rank() == 0){
// 			stdex::mdspan<int, stdex::dextents<int, 1>> view(v.data(),10);
// 			mgh.send_new(view, 1, 10, empi::Tag{1});
// 		}
// 		else {
// 			MPI_Status s;
// 			mgh.recv_new(v, 0, 10, empi::Tag{1}, s);
			
// 			for (int i = 0; i < 10; i++) {
// 				REQUIRE(v[i] == i);
// 			}
// 		}
// 	});
// }


// TEST_CASE("Send and receive a column of a matrix", "[mgh][layouts]"){
// 	empi::Context ctx{nullptr,nullptr};
// 	auto mg = ctx.create_message_group(MPI_COMM_WORLD);

// 	std::vector<int> v(16);
// 	if(mg->rank() == 0)
// 		std::iota(v.begin(), v.end(), 0);

// 	mg->run([&](empi::MessageGroupHandler<float>& mgh){
// 		if(mg->rank() == 0){
// 			auto view = empi::layouts::column_layout::build(v, stdex::dextents<int, 2>(4,4), 3);
// 			mgh.send_new(view, 1, 4, empi::Tag{1});
// 		}
// 		else {
// 			MPI_Status s;
// 			std::vector<int> dest(4);
// 			mgh.recv_new(dest, 0, 4, empi::Tag{1}, s);
			
// 			for (int i = 0; i < 4; i++) {
// 				REQUIRE(dest[i] == i*4+3);
// 			}
// 		}
// 	});
// }

// TEST_CASE("Send and receive a vector of struct", "[mgh][layouts]"){
// 	empi::Context ctx{nullptr,nullptr};
// 	auto mg = ctx.create_message_group(MPI_COMM_WORLD);

// 	struct S{
// 		int x,y;
// 	};

// 	std::vector<S> v(16);
// 	if(mg->rank() == 0)
// 		std::transform(v.begin(), v.end(), v.begin(), [i=0](S& s) mutable { s.x = i++; s.y = i++; return s;});
	
// 	mg->run([&](empi::MessageGroupHandler<float>& mgh){
// 		if(mg->rank() == 0){
// 			auto view = empi::layouts::contiguous_layout::build(v, empi::layouts::struct_layout::struct_accessor<S>());
// 			for (int i = 0; i < 16; i++) {
// 				REQUIRE(view(i).x == i*2); REQUIRE(view(i).y == i*2+1);
// 			}
// 			mgh.send_new(view, 1, 16, empi::Tag{1});
// 		}
// 		else {
// 			MPI_Status s;
// 			std::vector<S> dest(16);
// 			mgh.recv_new(dest, 0, 16, empi::Tag{1}, s);
			
// 			for (int i = 0; i < 16; i++) {
// 				REQUIRE(dest[i].x == i*2); REQUIRE(dest[i].y == i*2+1);
// 			}
// 		}
// 	});
// }

// TEST_CASE("Send and receive a vector of one field of a struct", "[mgh][layouts]"){
// 	empi::Context ctx{nullptr,nullptr};
// 	auto mg = ctx.create_message_group(MPI_COMM_WORLD);

// 	struct S{
// 		int x,y;
// 	};

// 	std::vector<S> v(16);
// 	if(mg->rank() == 0)
// 		std::transform(v.begin(), v.end(), v.begin(), [i=0](S& s) mutable { s.x = i++; s.y = i++; return s;});
	
// 	mg->run([&](empi::MessageGroupHandler<float>& mgh){
// 		if(mg->rank() == 0){
// 			auto proj = [](S& s) -> int& {return s.x;};
// 			auto view = empi::layouts::contiguous_layout::build(v, empi::layouts::struct_layout::struct_accessor<S, decltype(proj)>(std::move(proj)));
// 			for (int i = 0; i < 16; i++) {
// 				REQUIRE(view(i) == i*2);
// 			}
// 			mgh.send_new(view, 1, 16, empi::Tag{1});
// 		}
// 		else {
// 			MPI_Status s;
// 			std::vector<int> dest(16);
// 			mgh.recv_new(dest, 0, 16, empi::Tag{1}, s);
			
// 			for (int i = 0; i < 16; i++) {
// 				REQUIRE(dest[i] == i*2);
// 			}
// 		}
// 	});
// }

// TEST_CASE("Send and receive a vector of a subest of fields of a struct", "[mgh][layouts]"){
// 	empi::Context ctx{nullptr,nullptr};
// 	auto mg = ctx.create_message_group(MPI_COMM_WORLD);

// 	struct S{
// 		int x,y,z;
// 	};

// 	std::vector<S> v(16);
// 	if(mg->rank() == 0)
// 		std::transform(v.begin(), v.end(), v.begin(), [i=0](S& s) mutable { s.x = i++; s.y = i++, s.z = i++; return s;});
	
// 	mg->run([&](empi::MessageGroupHandler<float>& mgh){
// 		if(mg->rank() == 0){
// 			auto proj = [](S& s) -> std::tuple<int,int> {return {s.x,s.z};};
// 			auto view = empi::layouts::contiguous_layout::build(v, empi::layouts::struct_layout::struct_accessor<S, decltype(proj)>(std::move(proj)));
// 			for (int i = 0; i < 16; i++) {
// 				REQUIRE(std::get<0>(view(i)) == i*3);
// 				REQUIRE(std::get<1>(view(i)) == i*3 + 2);
// 			}
// 			mgh.send_new(view, 1, 16, empi::Tag{1});
// 		}
// 		else {
// 			MPI_Status s;
// 			std::vector<std::tuple<int,int>> dest(16);
// 			mgh.recv_new(dest, 0, 16, empi::Tag{1}, s);
			
// 			for (int i = 0; i < 16; i++) {
// 				REQUIRE(std::get<0>(dest[i]) == i*3);
// 				REQUIRE(std::get<1>(dest[i]) == i*3 + 2);
// 			}
// 		}
// 	});
// }

TEST_CASE("Send and receive a column of a subest of fields of a struct", "[mgh][layouts]"){
	empi::Context ctx{nullptr,nullptr};
	auto mg = ctx.create_message_group(MPI_COMM_WORLD);

	std::vector<trivial_struct> v(36);
	if(mg->rank() == 0){
		std::transform(v.begin(), v.end(), v.begin(), [i=0](trivial_struct& s) mutable { s.x = i++; s.y = i++, s.z = (static_cast<float>(i) * 1.2); i++; return s;});
	}

	mg->run([&](empi::MessageGroupHandler<float>& mgh){
		if(mg->rank() == 0){
			stdex::extents<size_t, 6,6> ext;
			constexpr int col = 0;
			auto acc = empi::layouts::make_struct_accessor<trivial_struct>(STRUCT_FIELDS(trivial_struct,x,z));
			auto view = empi::layouts::column_layout::build(v, ext, col,acc);
			for (int i = 0; i < 6; i++) {
				REQUIRE(std::get<0>(view(i)) == i * 3 * 6 + col * 3);
				REQUIRE(std::get<1>(view(i)) == Catch::Approx((i * 3 * 6 + col * 3 + 2) * 1.2));
			}
			mgh.send_new(view, 1, 6, empi::Tag{1});
		}
		else {
			MPI_Status s;
			std::vector<std::tuple<int,float>> dest(6);
			mgh.recv_new(dest, 0, 6, empi::Tag{1}, s);

			for (int i = 0; i < 6; i++) {
				REQUIRE(std::get<0>(dest[i]) == i*3 * 6);
				REQUIRE(std::get<1>(dest[i]) == Catch::Approx((i*3 * 6 + 2) * 1.2));
			}
		}
	});
}