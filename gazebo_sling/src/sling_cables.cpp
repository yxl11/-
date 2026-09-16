// Massless unilateral compliant cables. Gazebo integrates every rigid body.
// No pose/velocity setters or reference signals are used by this plugin.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <functional>

namespace gazebo {
class SlingCables : public WorldPlugin {
  physics::WorldPtr world;
  event::ConnectionPtr connection;
  transport::NodePtr transportNode;
  transport::PublisherPtr visuals;
  std::unique_ptr<ros::NodeHandle> rosNode;
  ros::Publisher telemetry;
  physics::LinkPtr payload;
  std::array<physics::LinkPtr, 3> uavs;
  double length = 1.8, stiffness = 5000., damping = 35.;
  double radius = .25, anchorZ = .125, lastLog = -1.;
 public:
  void Load(physics::WorldPtr w, sdf::ElementPtr sdf) override {
    world = w;
    if (sdf->HasElement("length")) length = sdf->Get<double>("length");
    if (sdf->HasElement("stiffness")) stiffness = sdf->Get<double>("stiffness");
    if (sdf->HasElement("damping")) damping = sdf->Get<double>("damping");
    if (!ros::isInitialized()) {
      int argc = 0; char **argv = nullptr;
      ros::init(argc, argv, "sling_cables", ros::init_options::NoSigintHandler);
    }
    rosNode.reset(new ros::NodeHandle("/sling"));
    telemetry = rosNode->advertise<std_msgs::Float64MultiArray>("cables", 2);
    transportNode.reset(new transport::Node()); transportNode->Init(world->Name());
    visuals = transportNode->Advertise<msgs::Visual>("~/visual");
    connection = event::Events::ConnectWorldUpdateBegin(std::bind(&SlingCables::Update, this));
  }
  void Update() {
    if (!payload) {
      auto m = world->ModelByName("payload");
      if (!m) return;
      payload = m->GetLink("body");
      for (int i = 0; i < 3; ++i) {
        auto u = world->ModelByName("uav" + std::to_string(i + 1));
        if (!u) { payload.reset(); return; }
        uavs[i] = u->GetLink("base_link");
      }
    }
    double t = world->SimTime().Double();
    bool log = t - lastLog >= .0199;
    std_msgs::Float64MultiArray data; data.data.push_back(t);
    auto pose = payload->WorldPose();
    for (int i = 0; i < 3; ++i) {
      double angle = 2 * M_PI * i / 3;
      ignition::math::Vector3d anchor(radius * cos(angle), radius * sin(angle), anchorZ);
      auto arm = pose.Rot().RotateVector(anchor);
      auto a = pose.Pos() + arm;
      auto b = uavs[i]->WorldCoGPose().Pos();
      auto delta = b - a; double distance = delta.Length();
      auto q = delta / std::max(distance, 1e-9);
      auto va = payload->WorldLinearVel() + payload->WorldAngularVel().Cross(arm);
      double speed = (uavs[i]->WorldLinearVel() - va).Dot(q);
      // Slack cables transmit no force; damping can never produce compression.
      double tension = distance > length ? std::max(0., stiffness * (distance - length) + damping * speed) : 0.;
      payload->AddForceAtWorldPosition(tension * q, a);
      uavs[i]->AddForceAtWorldPosition(-tension * q, b);
      data.data.insert(data.data.end(), {tension, distance, speed});
      if (log) {
        msgs::Visual visual;
        visual.set_name("payload::body::rope_" + std::to_string(i + 1));
        visual.set_parent_name("payload::body"); visual.set_cast_shadows(false);
        auto cylinder = visual.mutable_geometry()->mutable_cylinder();
        visual.mutable_geometry()->set_type(msgs::Geometry::CYLINDER);
        cylinder->set_radius(.009); cylinder->set_length(distance);
        ignition::math::Quaterniond rotation;
        rotation.From2Axes(ignition::math::Vector3d::UnitZ, q);
        msgs::Set(visual.mutable_pose(), pose.Inverse() * ignition::math::Pose3d((a + b) / 2., rotation));
        msgs::Set(visual.mutable_material()->mutable_diffuse(), ignition::math::Color(.85, .55, .12, 1));
        msgs::Set(visual.mutable_material()->mutable_ambient(), ignition::math::Color(.85, .55, .12, 1));
        visuals->Publish(visual);
      }
    }
    if (log) { telemetry.publish(data); lastLog = t; }
  }
};
GZ_REGISTER_WORLD_PLUGIN(SlingCables)
}
